# ===================== nullmodel, STREAMING numpy family -- the production inference path =====================
# The trained GRU mixture-density network from research_null_model.py, evaluated one step at a time in numpy
# (no torch at inference). Reproduces the research batch computation of the 17 nm_* features: max abs difference
# 2e-13 against the research torch path with the same (float64) weights on 8 synthetic series.
# Speed (this sandbox): history warm-up 24 ms per series (1,024 GRU steps), 66 us per online step.
#   * the warm-up projects each GRU layer's inputs for ALL history steps in one matrix product and runs only the
#     recurrent 192x64 product step by step;
#   * gate arithmetic is fused (one sigmoid over the r and z gates together, expit ufunc);
#   * per-step bookkeeping uses plain Python floats -- faster than numpy on 3-element arrays.
# Weights come from set_state({"weights": torch_state_dict_as_numpy}); init() raises if they were never set.
import math
import numpy as np
from scipy.special import ndtri as _nm_ndtri, erf as _nm_erf, expit as _nm_expit

class NullModelFamily(FeatureFamily):
    """Learned null model, streaming numpy implementation (no torch at inference).

    Needs the trained weights via set_state({"weights": {name: ndarray}}) -- the torch state_dict of _NullGRU as
    numpy float64. Reproduces the research batch computation (17 features, same names and order).
    Speed: the history warm-up (up to 1,024 steps) projects the inputs of each GRU layer for ALL steps in one matrix
    product and runs only the recurrent 192x64 product step by step; each online step is one fused GRU step per layer
    plus a 5-component mixture evaluation."""
    name = "nullmodel"
    version = 1
    cost = "O1"
    channels = ("z",)
    WARM, BURN = 1024, 32
    AGG_W = (32, 128, 512)
    STATE = None

    @classmethod
    def set_state(cls, state):
        w = {k: np.asarray(v, dtype=np.float64) for k, v in state["weights"].items()}
        cls.STATE = dict(
            Win=w["inp.weight"], bin=w["inp.bias"],
            layers=[(w[f"gru.weight_ih_l{l}"], w[f"gru.bias_ih_l{l}"], w[f"gru.weight_hh_l{l}"], w[f"gru.bias_hh_l{l}"])
                    for l in range(2)],
            Wout=w["out.weight"], bout=w["out.bias"])

    def names(self):
        return [f"nm_{s}_{a}" for s in ("nll", "g", "g2") for a in ("w32", "w128", "w512", "ew", "exp")] + \
               ["nm_g2_cusup", "nm_g2_cusdn"]

    # ---------- GRU pieces (PyTorch equations: gates r, z, n; b_hn inside the reset product) ----------
    @staticmethod
    def _inp(Win, bin_, z):
        z = np.asarray(z, dtype=np.float64)
        X = np.stack([z, np.abs(z), z * z], axis=-1)
        return np.maximum(X @ Win.T + bin_, 0.0)

    @staticmethod
    def _run_layer(Wih, bih, Whh, bhh, X, h):
        """All steps of one layer: input projections in one product, recurrence step by step."""
        H = h.size
        GI = X @ Wih.T + bih                                    # (T, 3H): input side of every step at once
        GIrz, GIn = np.ascontiguousarray(GI[:, :2 * H]), np.ascontiguousarray(GI[:, 2 * H:])
        out = np.empty((X.shape[0], H))
        for i in range(X.shape[0]):
            gh = Whh @ h + bhh
            rz = _nm_expit(GIrz[i] + gh[:2 * H])
            n = np.tanh(GIn[i] + rz[:H] * gh[2 * H:])
            h = n + rz[H:] * (h - n)                            # = (1 - z) n + z h
            out[i] = h
        return out, h

    @staticmethod
    def _step(Wih, bih, Whh, bhh, x, h):
        H = h.size
        gi = Wih @ x + bih
        gh = Whh @ h + bhh
        rz = _nm_expit(gi[:2 * H] + gh[:2 * H])
        n = np.tanh(gi[2 * H:] + rz[:H] * gh[2 * H:])
        return n + rz[H:] * (h - n)

    def _mix(self, Hout):
        """mixture parameters from GRU outputs: log-weights, means, sds (rows = steps)."""
        O = Hout @ self.S["Wout"].T + self.S["bout"]
        K = O.shape[-1] // 3
        lg, mu, ls = O[..., :K], O[..., K:2 * K], np.clip(O[..., 2 * K:], -4.0, 2.0)
        m = lg.max(axis=-1, keepdims=True)
        lw = lg - m - np.log(np.exp(lg - m).sum(axis=-1, keepdims=True))
        return lw, mu, np.exp(ls), ls

    @staticmethod
    def _score(lw, mu, sd, ls, y):
        """nll and predictive CDF of y (vectorized over rows)."""
        y = np.asarray(y, dtype=np.float64)[..., None]
        lp = -0.5 * ((y - mu) / sd) ** 2 - ls - 0.5 * math.log(2 * math.pi)
        a = lw + lp
        m = a.max(axis=-1, keepdims=True)
        nll = -(m[..., 0] + np.log(np.exp(a - m).sum(axis=-1)))
        cdf = (np.exp(lw) * 0.5 * (1.0 + _nm_erf((y - mu) / (sd * math.sqrt(2.0))))).sum(axis=-1)
        return nll, np.clip(cdf, 1e-6, 1 - 1e-6)

    # ---------- family interface ----------
    def init(self, hist):
        if self.STATE is None:
            raise RuntimeError("nullmodel: set_state() was never called -- the trained weights are missing")
        self.S = S = self.STATE
        xs = _history(hist)
        srt = np.sort(xs)
        zh = _nm_ndtri((np.searchsorted(srt, xs, side="left") + 0.5) / (srt.size + 1.0))
        zw = zh[-self.WARM:]
        wh = zw.size
        X = self._inp(S["Win"], S["bin"], zw)
        hs = []
        for (Wih, bih, Whh, bhh) in S["layers"]:
            X, h_last = self._run_layer(Wih, bih, Whh, bhh, X, np.zeros(Whh.shape[1]))
            hs.append(h_last)
        self.h = hs
        lw, mu, sd, ls = self._mix(X)                           # predictions made after each history element
        nll, cdf = self._score(lw[:-1], mu[:-1], sd[:-1], ls[:-1], zw[1:])
        g = _nm_ndtri(cdf)
        streams = {"nll": nll, "g": g, "g2": g * g - 1.0}
        sl = slice(self.BURN, wh - 1)
        self.mu_s = {k: float(v[sl].mean()) for k, v in streams.items()}
        self.sd_s = {k: float(v[sl].std()) + 1e-6 for k, v in streams.items()}
        g2h = (streams["g2"][sl] - self.mu_s["g2"]) / self.sd_s["g2"]
        cu = cd = 0.0
        ups, dns = np.empty(g2h.size), np.empty(g2h.size)
        for i, v in enumerate(g2h):
            cu = max(0.0, cu + v - 0.5)
            cd = max(0.0, cd - v - 0.5)
            ups[i], dns[i] = cu, cd
        self.q_up = max(float(np.quantile(ups, 0.95)), 1.0) if ups.size else 1.0
        self.q_dn = max(float(np.quantile(dns, 0.95)), 1.0) if dns.size else 1.0
        self.pred = (lw[-1], mu[-1], sd[-1], ls[-1])            # prediction for online element 0
        self.n = 0
        self.mu_l = [self.mu_s[k] for k in ("nll", "g", "g2")]
        self.sd_l = [self.sd_s[k] for k in ("nll", "g", "g2")]
        self.CAP = max(self.AGG_W)
        self.buf = [[0.0] * self.CAP for _ in range(3)]          # plain Python lists: faster than numpy at this size
        self.wsum = [[0.0] * len(self.AGG_W) for _ in range(3)]
        self.ew = [0.0, 0.0, 0.0]
        self.tot = [0.0, 0.0, 0.0]
        self.cu = self.cd = 0.0

    def update(self, ctx):
        z = float(ctx.vals["z"])
        lw, mu, sd, ls = self.pred
        d = (z - mu) / sd
        a = lw - 0.5 * d * d - ls
        m = float(a.max())
        nll = -(m + math.log(float(np.exp(a - m).sum())) - 0.5 * math.log(2 * math.pi))
        cdf = float(np.exp(lw) @ (0.5 * (1.0 + _nm_erf(d / math.sqrt(2.0)))))
        g = float(_nm_ndtri(min(max(cdf, 1e-6), 1 - 1e-6)))
        raw = (nll, g, g * g - 1.0)
        i = self.n % self.CAP
        self.n = n = self.n + 1
        bias = 1.0 - 0.98 ** n
        sq_ew, sq_n = math.sqrt(min(n, 100)), math.sqrt(n)
        out = []
        for s_ in range(3):
            v = (raw[s_] - self.mu_l[s_]) / self.sd_l[s_]
            buf, ws = self.buf[s_], self.wsum[s_]
            for j, W in enumerate(self.AGG_W):
                if n > W:
                    ws[j] -= buf[(i - W) % self.CAP]
                ws[j] += v
                m = W if n > W else n
                out.append(_safe(ws[j] / m * math.sqrt(m)))
            buf[i] = v
            self.ew[s_] = 0.02 * v + 0.98 * self.ew[s_]
            out.append(_safe(self.ew[s_] / bias * sq_ew))
            self.tot[s_] += v
            out.append(_safe(self.tot[s_] / n * sq_n))
            if s_ == 2:
                self.cu = max(0.0, self.cu + v - 0.5)
                self.cd = max(0.0, self.cd - v - 0.5)
        out += [_safe(self.cu / self.q_up), _safe(self.cd / self.q_dn)]
        # advance the GRU with the new value -> prediction for the next step
        x = self.S["Win"] @ np.array([z, abs(z), z * z]) + self.S["bin"]
        np.maximum(x, 0.0, out=x)
        for li, (Wih, bih, Whh, bhh) in enumerate(self.S["layers"]):
            x = self.h[li] = self._step(Wih, bih, Whh, bhh, x, self.h[li])
        O = self.S["Wout"] @ x + self.S["bout"]
        K = O.size // 3
        lg, mu, ls = O[:K], O[K:2 * K], np.minimum(np.maximum(O[2 * K:], -4.0), 2.0)
        mx = float(lg.max())
        lw = lg - mx - math.log(float(np.exp(lg - mx).sum()))
        self.pred = (lw, mu, np.exp(ls), ls)
        return out
