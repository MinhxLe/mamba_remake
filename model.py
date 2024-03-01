import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal
from jax.numpy.linalg import eigh, matrix_power
from jax.scipy.signal import convolve


def causal_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


def scan_SSM(Ab, Bb, Cb, u, x0):
    # no D b/c does not interact with input state
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def K_conv(Ab, Bb, Cb, L):
    return np.array([(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)])


# == SSM ==


class _SSMLayer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # SSM parameters
        self.A = self.param("A", lecun_normal(), (self.N, self.N))
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        self.K = K_conv(self.A, self.B, self.C, self.l_max)

        # RNN cache for long sequences
        # self.x_k_1 = self.variable("cache", "cache_x_k", np.zeros, (self.N,))

    def __call__(self, u):
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(
                self.A, self.B, self.C, u[:, np.newaxis], self.x_k_1.value
            )
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


def cloneLayer(layer):
    # used to have a layer per feature
    return nn.vmap(
        layer,
        in_axes=1,  # over feature dimension
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


# for feature dimension
SSMLayer = cloneLayer(_SSMLayer)


# == S4 ==


# Factory for constant initializer in Flax
def init(x):
    # Flax Module declare params as
    # A = self.param("A", init_func, shape)
    # init_func takes a prng key and rest of the args from param (usually shape)
    # So this init function will create the param with value (const) x
    def _init(key, shape):
        assert shape == x.shape
        return x

    return _init


def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return A


def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A = make_HiPPO(N)
    # Q = P , so only P is needed
    P = np.sqrt(np.arange(N) + 0.5)
    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)

    # S[i, j] = -0.5                     if i=j
    #         =  sqrt((2i+1)*(2j+1))/2   if j < j
    #         = -sqrt((2i+1)*(2j+1))/2  if j > j
    # So S is skew-symmetry (S^* = -S), and thus S is normal
    # https://en.wikipedia.org/wiki/Normal_matrix
    S = -A + P[:, np.newaxis] * P[np.newaxis, :]

    # todo: S_diag is constant (-0.5), not sure why we take mean here
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # (1) Diagonalize S to V \Lambda V^*
    #  -> V is columns of eigenvectors of S, Lambda is diagonal of eigenvalues
    # (2) We want to apply eigh: getting eigenvectors of Hermitian matrix.
    #     Hermitian matrix: equals to its conjugate transpose
    #     eigh is better (e.g. GPU support) than vanilla eig
    # (3) S is skew-sym, so (S without diagonal part) * i is Hermitian
    S_wo_diagonal = S - np.diag(np.diagonal(S))
    Lambda_imag, V = eigh(S_wo_diagonal * -1j)
    # With this:
    # (1) V @ diag(Lambda_imag) @ V^* = S_wo_diagonal * -1j (what eigh was for)
    # (2) Lambda_real is -0.5 * I, so V @ diag(Lambda_real) @ V^* = -0.5 *I (V is unitary) = S's diagonal part
    # combine => V @ diag(Lambda + Lambda_imag * i) @ V^* = S
    assert np.allclose(
        S, V @ np.diag(Lambda_real + Lambda_imag * 1j) @ V.conj().T, atol=1e-3
    )

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V


# ===== Layer agnostic below (the model)


class SequenceBlock(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Hyperparameters of inner layer
    dropout: float
    d_model: int
    prenorm: bool = True
    glu: bool = True
    training: bool = True
    decode: bool = False

    def setup(self):
        self.seq = self.layer_cls(**self.layer, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        # gated linear unit
        if self.glu:
            self.out2 = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.drop(x)
        if not self.prenorm:
            x = self.norm(x)
        return x


class Embedding(nn.Embed):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x):
        y = nn.Embed(self.num_embeddings, self.features)(x[..., 0])
        return np.where(x > 0, y, 0.0)  # x is onehot


class StackedModel(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Extra arguments to pass into layer constructor
    d_output: int
    d_model: int
    n_layers: int
    prenorm: bool = True
    dropout: float = 0.0
    embedding: bool = False  # Use nn.Embed instead of nn.Dense encoder
    classification: bool = False
    training: bool = True
    decode: bool = False  # Probably should be moved into layer_args

    def setup(self):
        if self.embedding:
            self.encoder = Embedding(self.d_output, self.d_model)
        else:
            self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                layer=self.layer,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                x = x / 255.0  # Normalize
            if not self.decode:
                x = np.pad(x[:-1], [(1, 0), (0, 0)])
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.classification:
            x = np.mean(x, axis=0)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)
