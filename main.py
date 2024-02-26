from functools import partial
from data import Datasets

from model import BatchStackedModel, SSMLayer
import numpy as np
import jax

key = jax.random.PRNGKey(0)
key, rng, train_rng = jax.random.split(key, num=3)
trainloader, testloader, n_classes, l_max, d_input = Datasets["sin"]()

# initalizing the model
MODEL_CONFIG = dict(
    d_model=64,
    n_layers=10,
    dropout=0.0,
    embedding=False,
    layer=dict(
        E=1,  # this has to be input size??
        N=64,
        l_max=l_max,
    ),
)
model_cls = partial(
    BatchStackedModel,
    layer_cls=SSMLayer,
    d_output=n_classes,
    classification=False,
    **MODEL_CONFIG,
)
model = model_cls(training=True)
init_rng, dropout_rng = jax.random.split(rng, num=2)
init_data = np.array(
    next(iter(trainloader))[0].numpy()
)  # need shape of data to initialize
params = model.init(
    {"params": init_rng, "dropout": dropout_rng},  # rng initialization
    np.array(next(iter(trainloader))[0].numpy()),  # need shape of data to initialize
)
print(model.apply(params, init_data, rngs=dict(dropout=dropout_rng)))
