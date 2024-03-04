import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve


def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


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


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


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

        # Step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))
        step = np.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step=step)

        #self.K = K_conv(self.A, self.B, self.C, self.l_max)
        self.K = K_conv(*self.ssm, self.l_max)

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
    # So A is NPLR (normal S plus low-rank PP*)

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

    # So we have: 
    # -A = S - P P^*                  -> NPLR
    #    = V Lambda V^* - P P^*
    #    = V ( Lambda - Q Q^* ) V^*    where Q = V^* P, 
    # So A can be conjugated to Lambda - Q Q^*  -> DPLR diagonal plus low rank
    # Since SSM is equivalent under conjugation: A -> V^*AV, B -> V^* B
    # We have the DPLR SSM: 
    # A' = Lambda - Q Q^*   where Q = V^* P 
    # B' = V^* B

    P = V.conj().T @ P  # The Q above
    B = V.conj().T @ B  # B' 
    return Lambda_real + 1j * Lambda_imag, P, B, V


def hippo_initializer(N):
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return init(Lambda.real), init(Lambda.imag), init(P), init(B)


def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    """With DPLR structure, discretizing SSM doesn't need to do inverse
       because DPLR inverse is also DPLR

       => S4 recurrence is O(N) per step where N is state dim
    """
    # Convert parameters to matrices
    B = B[:, np.newaxis]
    # just learn Ct as the parameter
    Ct = C[np.newaxis, :]

    N = Lambda.shape[0]
    A = np.diag(Lambda) - P[:, np.newaxis] @ Q[:, np.newaxis].conj().T
    I = np.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = np.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


@jax.jit
def cauchy(v, omega, lambd):
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)


def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    """Given: DPLR SSM with A = Lambda - PQ^*
       Returns the conv kernel K_bar
 
       1) conv kernel K_bar can be efficiently computed by inverse FFT of
       truncated generating function (z-transform) of the kernel

       2) truncated (only L terms) generating function:
       \sigma_{i=0}^{L-1} C_b A_b^i B_b z^i = Ct (I - A_b z)^-1 B_bar
       has only one DPLR matrix inverse , no matrix power needed.
       DPLR inverse can be computed efficiently by:
       3) Apply Woodbury Identity: inverse of DPLR -> inverse of diagonal
       4) The result can be simplied to "Cauchy kernel", stable and fast

       => computing kernel is O(N+L)
    """
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = np.exp((-2j * np.pi) * (np.arange(L) / L))
    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = np.fft.ifft(atRoots, L).reshape(L)
    return out.real


class S4Layer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    # Special parameters with multiplicative factor on lr and no weight decay (handled by main train script)
    lr = {
        "Lambda_re": 0.1,
        "Lambda_im": 0.1,
        "P": 0.1,
        "B": 0.1,
        "log_step": 0.1,
    }

    def setup(self):
        print("in setup")
        # Learned Parameters (C is complex!)
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)
        self.Lambda_re = self.param("Lambda_re", init_A_re, (self.N,))
        self.Lambda_im = self.param("Lambda_im", init_A_im, (self.N,))
        # Ensure the real part of Lambda is negative
        # (described in the SaShiMi follow-up to S4)
        self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        self.P = self.param("P", init_P, (self.N,))
        self.B = self.param("B", init_B, (self.N,))
        # C should be init as standard normal
        # This doesn't work due to how JAX handles complex optimizers https://github.com/deepmind/optax/issues/196
        # self.C = self.param("C", normal(stddev=1.0, dtype=np.complex64), (self.N,))
        self.C = self.param("C", normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))

        if not self.decode:
            # CNN mode, compute kernel.
            self.K = kernel_DPLR(
                self.Lambda,
                self.P,
                self.P,
                self.B,
                self.C,
                self.step,
                self.l_max,
            )

        else:
            # RNN mode, discretize

            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return discrete_DPLR(
                    self.Lambda,
                    self.P,
                    self.P,
                    self.B,
                    self.C,
                    self.step,
                    self.l_max,
                )

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", np.zeros, (self.N,), np.complex64
            )
        print("done setup")

    def __call__(self, u):
        # This is identical to SSM Layer
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


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
