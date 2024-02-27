from functools import partial

import flax
import jax
import optax
from flax.training import train_state
from jax import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from data import Datasets
from model import BatchStackedModel, SSMLayer

DEBUG_MODE = True

dataset = "mnist-classification"
classification = ("classification" in dataset,)
key = jax.random.PRNGKey(0)
key, rng, train_rng = jax.random.split(key, num=3)
trainloader, testloader, n_classes, l_max, d_input = Datasets[dataset]()
# initalizing the model
if DEBUG_MODE:
    num_epochs = 1
    MODEL_CONFIG = dict(
        d_model=8,
        n_layers=2,
        dropout=0.0,
        embedding=False,
        layer=DictConfig(
            dict(
                N=8,
                l_max=l_max,
            )
        ),
    )
else:
    num_epochs = 10
    MODEL_CONFIG = dict(
        d_model=128,
        n_layers=64,
        dropout=0.0,
        embedding=False,
        layer=DictConfig(
            dict(
                N=64,
                l_max=l_max,
            )
        ),
    )
model_cls = partial(
    BatchStackedModel,
    layer_cls=SSMLayer,
    d_output=n_classes,
    classification=classification,
    **MODEL_CONFIG,
)

# =====================================


# As we're using Flax, we also write a utility function to return a default TrainState object.
# This function initializes model parameters, as well as our optimizer. Note that for S4 models,
# we use a custom learning rate for parameters of the S4 kernel (lr = 0.001, no weight decay).
def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(
    rng,
    model_cls,
    trainloader,
    lr=1e-3,
    lr_layer=None,
    lr_schedule=False,
    weight_decay=0.0,
    total_steps=-1,
):
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        np.array(next(iter(trainloader))[0].numpy()),
    )
    # Note: Added immediate `unfreeze()` to play well w/ Optax. See below!
    params = flax.core.unfreeze(params["params"])
    # TODO: why?

    if lr_schedule:
        schedule_fn = lambda lr: optax.cosine_onecycle_schedule(
            peak_value=lr,
            transition_steps=total_steps,
            pct_start=0.1,
        )
    else:
        schedule_fn = lambda lr: lr
    # lr_layer is a dictionary from parameter name to LR multiplier
    if lr_layer is None:
        lr_layer = {}

    optimizers = dict()
    optimizers["__default__"] = optax.adamw(
        learning_rate=schedule_fn(lr),
        weight_decay=weight_decay,
    )
    name_map = map_nested_fn(lambda k, _: k if k in lr_layer else "__default__")
    tx = optax.multi_transform(optimizers, name_map)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


@partial(jax.jit, static_argnums=(3, 4))
def eval_step(batch_inputs, batch_labels, params, model, classification=False):
    if not classification:
        batch_labels = batch_inputs[:, :, 0]
    logits = model.apply({"params": params}, batch_inputs)
    # TODO does cross entropy loss even make sense if it's not classification?
    loss = np.mean(cross_entropy_loss(logits, batch_labels))
    acc = np.mean(compute_accuracy(logits, batch_labels))
    return loss, acc


def validate(params, model, testloader, classification=False):
    # Compute average loss & accuracy
    model = model(training=False)
    losses, accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(tqdm(testloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())  # Not the most efficient...
        loss, acc = eval_step(
            inputs, labels, params, model, classification=classification
        )
        losses.append(loss)
        accuracies.append(acc)

    return np.mean(np.array(losses)), np.mean(np.array(accuracies))


# @partial(jax.jit, static_argnums=(4, 5))
def train_step(state, rng, batch_inputs, batch_labels, model, classification=False):
    def loss_fn(params):
        logits, mod_vars = model.apply(
            {"params": params},
            batch_inputs,
            rngs={"dropout": rng},
            mutable=["intermediates"],
        )
        loss = np.mean(cross_entropy_loss(logits, batch_labels))
        acc = np.mean(compute_accuracy(logits, batch_labels))
        return loss, (logits, acc)

    if not classification:
        batch_labels = batch_inputs[:, :, 0]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, acc)), grads = grad_fn(state.params)
    print(loss)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc


state = create_train_state(
    rng,
    model_cls,
    trainloader,
    lr=5e-3,
)
for epoch in range(num_epochs):
    print(f"[*] Starting Training Epoch {epoch + 1}...")
    model = model_cls(training=True)
    batch_losses, batch_accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())
        rng, drop_rng = jax.random.split(rng)
        state, loss, acc = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            model,
            classification=classification,
        )
        batch_losses.append(loss)
        batch_accuracies.append(acc)

    print(f"[*] Running Epoch {epoch + 1} Validation...")
    test_loss, test_acc = validate(
        state.params, model_cls, testloader, classification=classification
    )
