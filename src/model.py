import numpy as np
from copy import copy
from numba import njit
from numba.experimental import jitclass
from numba import float64, int64, boolean
from numba.extending import register_jitable

from vbi.models.numba.utils import check_vec_size_1d, check_parameters

# ----------------------------------------
# Reduced Wong–Wang spec (excitatory only)
# ----------------------------------------

rww_spec = [
    ("a", float64),
    ("b", float64),
    ("d", float64),
    ("tau_s", float64),
    ("gamma", float64),
    ("w", float64[:]), # changed from float64 -- now each region can have its w
    ("J", float64), 
    ("G", float64),
    ("I_ext", float64[:]),
    ("weights", float64[:, :]),
    ("sigma", float64),
    ("dt", float64),
    ("t_end", float64),
    ("t_cut", float64),
    ("nn", int64),
    ("seed", int64),
    ("initial_state", float64[:]),
    ("RECORD_S", boolean),
    ("s_decimate", int64),
]


@jitclass(rww_spec)
class ParRWW:
    def __init__(
        self,
        a=270.0/1000.0,
        b=108.0/1000.0,
        d=0.154 * 1000.0,
        tau_s=100.0,  # ms
        gamma=0.641,
        # a=270.0/ 1000.0,
        # b=108.0/ 1000.0,
        # d=0.154* 1000.0,
        # tau_s=100.0/ 1000.0,  # ms
        # gamma=0.641,
        # gamma=0.9,  
        w=np.array([0.9]),
        J=0.2609,
        G=0.0,
        I_ext=np.array([0.382]),
        weights=np.array([[0.0]]),
        sigma=0.05,
        dt=1.0,
        t_end=1000.0,
        t_cut=0.0,
        nn=1,
        seed=-1,
        initial_state=np.array([0.0]),
        RECORD_S=True,
        s_decimate=1,
    ):
        self.a = a
        self.b = b
        self.d = d
        self.tau_s = tau_s
        self.gamma = gamma
        self.w = w
        self.J = J
        self.G = G
        self.I_ext = I_ext
        self.weights = weights
        self.sigma = sigma
        self.dt = dt
        self.t_end = t_end
        self.t_cut = t_cut
        self.nn = nn
        self.seed = seed
        self.initial_state = initial_state
        self.RECORD_S = RECORD_S
        self.s_decimate = s_decimate


# -----------------------------
# Transfer function H(x)
# -----------------------------
@njit
def H(x, a, b, d):
    u = a * x - b
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        den = 1.0 - np.exp(-d * u[i])
        if np.abs(den) < 1e-12:
            out[i] = a * u[i] * 0.5
        else:
            out[i] = u[i] / den
    return out


# -----------------------------
# RHS of reduced model
# -----------------------------
@njit
def f_rww(S, t, P):
    x = P.w * P.J * S + P.G * P.J * P.weights.dot(S) + P.I_ext
    r = H(x, P.a, P.b, P.d)
    dSdt = -S / P.tau_s + (1.0 - S) * P.gamma * r
    return dSdt

# -----------------------------
# RHS of reduced model - all outputs
# -----------------------------
@njit
def f_rww_full(S, t, P):
    x = P.w * P.J * S + P.G * P.J * P.weights.dot(S) + P.I_ext

    p1 = P.w * P.J * S
    p2 = P.G * P.J * P.weights.dot(S)
    p3 = P.I_ext  # already length nn (after check_vec_size_1d)

    r = H(x, P.a, P.b, P.d)
    dSdt = -S / P.tau_s + (1.0 - S) * P.gamma * r

    return dSdt, x, r, p1, p2, p3

# -----------------------------
# Heun SDE integrator
# -----------------------------
@njit
def heun_sde(S, t, P):
    dt = P.dt
    nn = P.nn
    dW = P.sigma * np.sqrt(dt) * np.random.randn(nn)
    k1 = f_rww(S, t, P)
    y_ = S + dt * k1 + dW
    k2 = f_rww(y_, t + dt, P)
    return S + 0.5 * dt * (k1 + k2) + dW


# -----------------------------
# Public API class (NO BOLD)
# -----------------------------
class RWW_sde:
    def __init__(self, par: dict = {}) -> None:
        self.valid_par = [rww_spec[i][0] for i in range(len(rww_spec))]
        check_parameters(par, self.valid_par, rww_spec)

        nn = par.get("nn", None)
        weights = par.get("weights", None)
        if weights is None:
            weights = np.zeros((1, 1), dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        if nn is None:
            nn = weights.shape[0]
        par.setdefault("nn", nn)

        seed = par.get("seed", -1)
        if seed > 0:
            set_seed_compat(seed)

        # initial state
        if "initial_state" in par:
            par["initial_state"] = np.array(par["initial_state"], dtype=np.float64)
        else:
            par["initial_state"] = set_initial_state(nn, seed=seed)

        # --- RWW parameters ---
        self.P = ParRWW()
        for k, v in par.items():
            if k == "I_ext":
                v = check_vec_size_1d(v, nn)
            if k == "w":
                v = check_vec_size_1d(v, nn)
            setattr(self.P, k, v)

    def run(self, par: dict = {}, x0=None):
        if x0 is None:
            S = copy(self.P.initial_state)
        else:
            S = np.array(x0, dtype=np.float64)

        if self.P.seed > 0:
            set_seed_compat(self.P.seed)
            reset_numba_rng(self.P.seed)  # keep behavior consistent

        check_parameters(par, self.valid_par, rww_spec)

        if par:
            for k, v in par.items():
                if k == "I_ext":
                    v = check_vec_size_1d(v, self.P.nn)
                elif k == "w":
                    v = check_vec_size_1d(v, self.P.nn) 
                setattr(self.P, k, v)

        # --- sanity checks ---
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert len(S) == self.P.nn, "x0 must be length nn"
        assert self.P.t_cut < self.P.t_end

        # --- time grid ---
        nt = int(np.floor(self.P.t_end / self.P.dt))
        t = np.arange(nt) * self.P.dt
        valid_mask = t > self.P.t_cut
        s_buffer_len = int(np.sum(valid_mask) // max(1, self.P.s_decimate))

        t_buf = np.zeros((s_buffer_len,), dtype=np.float32)
        S_rec = (
            np.zeros((s_buffer_len, self.P.nn), dtype=np.float32)
            if self.P.RECORD_S
            else np.array([])
        )
        X_rec = np.zeros((s_buffer_len, self.P.nn), dtype=np.float32)
        R_rec = np.zeros((s_buffer_len, self.P.nn), dtype=np.float32)
        P1_rec = np.zeros((s_buffer_len, self.P.nn), dtype=np.float32)
        P2_rec = np.zeros((s_buffer_len, self.P.nn), dtype=np.float32)
        P3_rec = np.zeros((s_buffer_len, self.P.nn), dtype=np.float32)

        # --- main loop ---
        s_idx=0
        for i in range(nt):
            t_curr = i * self.P.dt

            dsdt, x, r, p1, p2, p3 = f_rww_full(S, t_curr, self.P)
            S = heun_sde(S, t_curr, self.P)

            if (t_curr > self.P.t_cut) and (i % max(1, self.P.s_decimate) == 0):
                if s_idx < s_buffer_len:
                    t_buf[s_idx] = t_curr
                    if self.P.RECORD_S:
                        S_rec[s_idx] = S
                    X_rec[s_idx] = x
                    R_rec[s_idx] = r
                    P1_rec[s_idx] = p1
                    P2_rec[s_idx] = p2
                    P3_rec[s_idx] = p3
                    s_idx += 1

        return {
            "S": S_rec,
            "t": t_buf,
            "X": X_rec,
            "R": R_rec,
            "P1": P1_rec,
            "P2": P2_rec,
            "P3": P3_rec,
        }


# -----------------------------
# API helpers
# -----------------------------

@njit
def set_initial_state(nn, seed=-1):
    if seed is not None and seed >= 0:
        set_seed_compat(seed)
    y0 = np.random.rand(nn) * 0.1
    return y0.astype(np.float64)


@register_jitable
def set_seed_compat(x):
    np.random.seed(x)


@njit
def reset_numba_rng(seed):
    np.random.seed(seed)
