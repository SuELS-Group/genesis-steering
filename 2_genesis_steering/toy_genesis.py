"""The genesis question in an idealized world. CPU only, seconds to minutes.

Part A  (no training) Theorem 1's own setting: an elliptical density whose two classes are split by a hyperplane
        orthogonal to one principal axis. Compares the steering displacement of the difference of class means of that
        distribution ("genesis") with vectors from other datasets of the same behaviour, and repeats with the boundary
        tilted off the principal axes (the assumption broken).
Part B  (a small MLP we train) the behaviour is the model's output; vectors are differences of class means of hidden
        activations; steering adds alpha * unit vector to the hidden layer. Three training regimes:
          genesis-only          the model learns its boundary from the genesis data alone
          pretrain + fine-tune  a large pretraining set with a related but different boundary, then genesis
                                fine-tuning of varying length (Tülu-like: most of the boundary predates the safety data)
          memorization          a small genesis set, a wide model trained to zero loss; vectors from members vs from
                                fresh samples of the same distribution (the toy version)
"""
from __future__ import annotations

import numpy as np

unit = lambda v: v / np.linalg.norm(v)


# =============================== Part A: the theorem's own world =====================================
def random_sigma(d, rng, lo=0.1, hi=10.0):
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = np.exp(np.linspace(np.log(lo), np.log(hi), d))
    return U, lam


def sample_elliptical(n, U, lam, rng, mu=None, kind="gauss", nu=3.0):
    z = rng.standard_normal((n, len(lam))) * np.sqrt(lam)
    if kind == "t":                                      # multivariate t: a Gaussian scaled by one chi2 draw per point
        z = z / np.sqrt(rng.chisquare(nu, size=(n, 1)) / nu)
    x = z @ U.T
    return x if mu is None else x + mu


def class_draw(X, y, n, rng):
    i = rng.choice(np.flatnonzero(y), n, replace=False); j = rng.choice(np.flatnonzero(~y), n, replace=False)
    return X[i], X[j]


def crossing_ratio(q, normal, H):
    """Median over points H (all on the negative side) of the displacement along unit q needed to reach the boundary,
    divided by the shortest possible (the distance along the normal) = 1 / |cos(q, normal)| for a hyperplane."""
    c = abs(unit(q) @ unit(normal))
    return np.inf if c < 1e-9 else 1.0 / c


def theorem_world(seed, d=16, k_axis=6, tilt=0.0):
    """(rng, U, lam, boundary normal, threshold, top-variance axis) of one Part A world."""
    rng = np.random.default_rng(seed)
    U, lam = random_sigma(d, rng)
    uk, utop = U[:, k_axis], U[:, -1]
    normal = unit(np.cos(tilt) * uk + np.sin(tilt) * utop)
    tau = 0.3 * np.sqrt(normal @ (U * lam) @ U.T @ normal)
    return rng, U, lam, normal, tau, utop


def part_a(seed, d=16, k_axis=6, kind="gauss", tilt=0.0, n=4000, n_draw=128):
    """One world. Returns {vector: (cos to the boundary normal, crossing-displacement ratio)}.
    tilt = 0: the boundary is orthogonal to principal axis k (the theorem). tilt > 0: its normal is rotated by that
    angle (radians) towards the largest-variance axis (the assumption broken)."""
    rng, U, lam, normal, tau, utop = theorem_world(seed, d, k_axis, tilt)
    label = lambda X, w=normal, t=tau: X @ w > t
    X = sample_elliptical(n, U, lam, rng, kind=kind); y = label(X)
    out = {}
    a, b = class_draw(X, y, n_draw, rng); g = a.mean(0) - b.mean(0)
    out["genesis: mean difference"] = g
    Sw = np.cov(np.r_[a - a.mean(0), b - b.mean(0)].T)
    out["genesis: LDA (Σw⁻¹ · difference)"] = np.linalg.solve(Sw + 1e-3 * np.eye(d), g)
    # external datasets of the same behaviour
    R, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Xr = sample_elliptical(n, R @ U, lam, rng, kind=kind); yr = label(Xr)                     # other input density
    a, b = class_draw(Xr, yr, n_draw, rng); out["external: other density"] = a.mean(0) - b.mean(0)
    r = rng.standard_normal(d); r = unit(r - (r @ normal) * normal)
    w2 = unit(normal + 0.5 * r)
    y2 = label(X, w2, tau)                                                                     # slightly other rule
    a, b = class_draw(X, y2, n_draw, rng); out["external: related rule"] = a.mean(0) - b.mean(0)
    Xs = sample_elliptical(n, U, lam, rng, mu=1.5 * np.sqrt(lam[-1]) * utop, kind=kind); ys = label(Xs)
    a, b = class_draw(Xs, ys, n_draw, rng); out["external: shifted density"] = a.mean(0) - b.mean(0)
    out["random direction"] = rng.standard_normal(d)
    return {name: (abs(unit(v) @ normal), crossing_ratio(v, normal, None)) for name, v in out.items()}


# =============================== Part B: a small MLP ==================================================
def _torch():
    import torch
    torch.set_num_threads(8)
    return torch


class World:
    """Input distributions and labelling rules of one Part B world (dimension D)."""

    def __init__(self, seed, D=20, rho_rule=0.85, rho_pre=0.8):
        rng = np.random.default_rng(seed); self.rng = rng; self.D = D
        self.w = unit(rng.standard_normal(D))                                       # the genesis behaviour
        def related(rho):
            r = rng.standard_normal(D); r = unit(r - (r @ self.w) * self.w)
            return unit(rho * self.w + np.sqrt(1 - rho ** 2) * r)
        self.w_rule, self.w_pre = related(rho_rule), related(rho_pre)
        def dens():
            U, lam = random_sigma(D, rng, 0.2, 5.0); return U, lam, 0.7 * rng.standard_normal(D)
        self.dens = {k: dens() for k in ("genesis", "other", "pre", "eval")}

    def confounded(self, n, strength=4.0):
        """Genesis inputs and labels, but selected so that a high-variance nuisance direction s (orthogonal to w)
        correlates with the label: the toy version of a dataset whose refusals also differ in style or topic."""
        U, lam, mu = self.dens["genesis"]
        s = U[:, -1] - (U[:, -1] @ self.w) * self.w; s = s / np.linalg.norm(s)
        X, y = self.sample("genesis", 4 * n)
        z = (X - X.mean(0)) @ s / np.std(X @ s)
        keep = self.rng.random(len(y)) < 1 / (1 + np.exp(-strength * z * np.where(y, 1, -1)))
        X, y = X[keep][:n], y[keep][:n]
        return X, y

    def sample(self, which, n, rule="genesis"):
        U, lam, mu = self.dens[which]
        X = sample_elliptical(n, U, lam, self.rng, mu=mu)
        w = {"genesis": self.w, "related": self.w_rule, "pre": self.w_pre}[rule]
        b = np.median(X @ w)                                                         # balanced classes
        return X.astype(np.float32), (X @ w > b)


def make_mlp(D, H, torch):
    nn = torch.nn
    return nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1))


def train(model, X, y, epochs, torch, lr=1e-3, bs=256, seed=0, wd=0.0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(X), torch.tensor(y.astype(np.float32))
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for s in range(0, len(Xt), bs):
            i = perm[s:s + bs]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xt[i]).squeeze(-1), yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def hidden(model, X, torch):
    """The steering site: the output of the first hidden layer (after its ReLU)."""
    with torch.no_grad():
        return model[1](model[0](torch.tensor(X))).numpy()


def head(model, Hh, torch):
    with torch.no_grad():
        return model[4](model[3](model[2](torch.tensor(Hh, dtype=torch.float32)))).squeeze(-1).numpy()


def boundary_normal(model, Heval, torch):
    """Mean gradient of the logit w.r.t. the steering site over evaluation points near the model's boundary."""
    Hh = torch.tensor(Heval, dtype=torch.float32, requires_grad=True)
    logit = model[4](model[3](model[2](Hh))).squeeze(-1)
    near = logit.detach().abs() < 1.0
    logit[near].sum().backward() if near.any() else logit.sum().backward()
    return unit(Hh.grad[near].mean(0).numpy() if near.any() else Hh.grad.mean(0).numpy())


def steer_success(model, Heval, q, doses, torch):
    """Both poles, as in the real experiments: points the model calls positive pushed by -alpha*q, points it calls
    negative pushed by +alpha*q; success at a dose = share that flips; S = mean over doses and the two poles."""
    q = unit(q); base = head(model, Heval, torch) > 0
    flips = []
    for sign, sel in ((-1, base), (+1, ~base)):
        H0 = Heval[sel]
        flips.append([np.mean((head(model, H0 + sign * a * q, torch) > 0) != (sign < 0)) for a in doses])
    return float(np.mean(flips)), np.array(flips)


def vectors(model, sets, torch, n_draw=128, rng=None):
    """{name: difference of class means of hidden activations}, 128 + 128 per set by the set's own labels."""
    out = {}
    for name, (X, y) in sets.items():
        a, b = class_draw(X, y, n_draw, rng)
        out[name] = hidden(model, a, torch).mean(0) - hidden(model, b, torch).mean(0)
    return out


def build_regime(seed, regime, ft_epochs=20, H=64, D=20):
    """Train one regime. Returns a dict with the model, the world, the vector sets, the evaluation activations, the
    model's boundary normal and the rng (consumed in the same order as before the refactor)."""
    torch = _torch(); torch.manual_seed(seed)
    W = World(seed, D=D); rng = np.random.default_rng(seed + 1)
    Xe, _ = W.sample("eval", 4000)
    if regime == "genesis-only":
        Xg, yg = W.sample("genesis", 4000); model = train(make_mlp(D, H, torch), Xg, yg, 30, torch, seed=seed)
        members = (Xg, yg)
    elif regime == "pretrain+finetune":
        Xp, yp = W.sample("pre", 40000, rule="pre")
        model = train(make_mlp(D, H, torch), Xp, yp, 10, torch, seed=seed)
        Xg, yg = W.sample("genesis", 2000)
        if ft_epochs: model = train(model, Xg, yg, ft_epochs, torch, seed=seed + 7)
        members = (Xg, yg)
    elif regime == "memorization":
        Xg, yg = W.sample("genesis", 300)
        model = train(make_mlp(D, 512, torch), Xg, yg, 1500, torch, lr=1e-3, bs=64, seed=seed)
        members = (Xg, yg)
    else:
        raise ValueError(regime)
    sets = {"genesis (training rows)": members,
            "genesis distribution, fresh rows": W.sample("genesis", 4000),
            "external: other density": W.sample("other", 4000),
            "external: related rule": W.sample("genesis", 4000, rule="related"),
            "external: confounded": W.confounded(4000)}
    if regime == "pretrain+finetune":
        sets["pretraining data"] = W.sample("pre", 4000, rule="pre")
    if regime == "memorization":                     # fresh rows: a new sample of the same size from the same distribution
        sets["genesis distribution, fresh rows"] = W.sample("genesis", 300)
    return {"torch": torch, "W": W, "rng": rng, "model": model, "members": members, "sets": sets, "Xe": Xe}


def run_regime(seed, regime, ft_epochs=20, H=64, D=20):
    """One seed of one regime. Returns rows {vector, S, cos to the model's boundary normal, ...}."""
    R = build_regime(seed, regime, ft_epochs, H, D)
    torch, rng, model, members, sets, Xe = R["torch"], R["rng"], R["model"], R["members"], R["sets"], R["Xe"]
    V = vectors(model, sets, torch, n_draw=min(128, int(min(members[1].sum(), (~members[1]).sum()))), rng=rng)
    He = hidden(model, Xe, torch); nrm = boundary_normal(model, He, torch)
    V["random direction"] = rng.standard_normal(He.shape[1])
    scale = np.median(np.linalg.norm(He, axis=1)); doses = scale * np.linspace(0.05, 1.0, 12)
    train_acc = float(np.mean((head(model, hidden(model, members[0], torch), torch) > 0) == members[1]))
    rows = []
    for name, v in V.items():
        S, _ = steer_success(model, He, v, doses, torch)
        rows.append({"seed": seed, "regime": regime, "ft_epochs": ft_epochs, "vector": name, "S": S,
                     "cos to boundary normal": float(abs(unit(v) @ nrm)), "train acc": train_acc})
    return rows


# =============================== Section C: why genesis data is not perfect ==========================
deg = lambda a, b: float(np.degrees(np.arccos(min(1.0, abs(unit(a) @ unit(b))))))


def shrunk_lda(Xa, Xb, lam=1e-2):
    """Sigma_w^{-1} (mean difference) with the within-class covariance shrunk towards its mean eigenvalue."""
    Z = np.r_[Xa - Xa.mean(0), Xb - Xb.mean(0)]; S = Z.T @ Z / (len(Z) - 2)
    t = np.trace(S) / len(S)
    return np.linalg.solve((1 - lam) * S + lam * t * np.eye(len(S)), Xa.mean(0) - Xb.mean(0))


def part_a_geometry(seed, tilt=0.0, kind="gauss", n_pop=200_000, n_draws=200, n_draw=128):
    """The theorem's world: the boundary normal, the population difference of means (its anisotropy bias), the
    population LDA direction, and n_draws difference-of-means vectors from 128 + 128 draws (sampling spread)."""
    rng, U, lam, normal, tau, utop = theorem_world(seed, tilt=tilt)
    X = sample_elliptical(n_pop, U, lam, rng, kind=kind); y = X @ normal > tau
    dom_pop = X[y].mean(0) - X[~y].mean(0)
    lda_pop = shrunk_lda(X[y][:50_000], X[~y][:50_000], lam=1e-4)
    draws = [np.subtract(*[m.mean(0) for m in class_draw(X, y, n_draw, rng)]) for _ in range(n_draws)]
    e2 = unit(utop - (utop @ normal) * normal) if tilt == 0 else unit(dom_pop - (dom_pop @ normal) * normal)
    return {"normal": normal, "tau": tau, "dom_pop": dom_pop, "lda_pop": lda_pop, "draws": draws, "e2": e2,
            "X": X[:1500], "y": y[:1500],
            "angles": {"sampling: 128-draw vs population mean difference": float(np.median([deg(d, dom_pop) for d in draws])),
                       "anisotropy: population mean difference vs boundary normal": deg(dom_pop, normal),
                       "total: 128-draw mean difference vs boundary normal": float(np.median([deg(d, normal) for d in draws])),
                       "population LDA vs boundary normal": deg(lda_pop, normal)}}


def input_normal(model, Xe, torch):
    """The model's boundary in INPUT space: mean gradient of the logit w.r.t. the input over evaluation points near it."""
    X = torch.tensor(Xe, requires_grad=True); logit = model(X).squeeze(-1)
    near = logit.detach().abs() < 1.0
    (logit[near] if near.any() else logit).sum().backward()
    return unit((X.grad[near] if near.any() else X.grad).mean(0).numpy())


def regime_geometry(seed, regime, ft_epochs=20, n_pop=20_000, n_draws=100, n_draw=128):
    """A trained model. In its hidden space (the steering site): the genesis vector's angle to the model's boundary
    normal, split into sampling (128-draw vs the genesis distribution's population mean difference) and the systematic
    part (population mean difference vs the model's normal); the same total for the pretraining data when present.
    In input space, where the rules are defined: the angle between the model's learned boundary and the rule that
    labels the genesis data (boundary mismatch), and the pretraining rule."""
    R = build_regime(seed, regime, ft_epochs)
    torch, model, W, rng = R["torch"], R["model"], R["W"], R["rng"]
    He = hidden(model, R["Xe"], torch); nrm = boundary_normal(model, He, torch)
    Xg, yg = W.sample("genesis", n_pop); Hg = hidden(model, Xg, torch)
    dom_pop = Hg[yg].mean(0) - Hg[~yg].mean(0)
    Xm, ym = R["members"]; Hm = hidden(model, Xm, torch)
    k = min(n_draw, int(ym.sum()), int((~ym).sum()))
    draws = [Hm[rng.choice(np.flatnonzero(ym), k, replace=False)].mean(0) - Hm[rng.choice(np.flatnonzero(~ym), k, replace=False)].mean(0)
             for _ in range(n_draws)]
    xn = input_normal(model, R["Xe"], torch)
    out = {"hidden: sampling (128-draw vs population)": float(np.median([deg(d, dom_pop) for d in draws])),
           "hidden: systematic (population vs model normal)": deg(dom_pop, nrm),
           "hidden: total (genesis 128-draw vs model normal)": float(np.median([deg(d, nrm) for d in draws])),
           "input: model boundary vs genesis rule": deg(xn, W.w)}
    extra = {}
    if "pretraining data" in R["sets"]:
        Xp, yp = W.sample("pre", n_pop, rule="pre"); Hp = hidden(model, Xp, torch)
        extra["pre_dom"] = Hp[yp].mean(0) - Hp[~yp].mean(0)
        out["hidden: total (pretraining population vs model normal)"] = deg(extra["pre_dom"], nrm)
        out["input: model boundary vs pretraining rule"] = deg(xn, W.w_pre)
    return {"angles": out, "model": model, "torch": torch, "He": He, "nrm": nrm, "dom_pop": dom_pop,
            "draws": draws, "Hm": Hm, "ym": ym, **extra}


def logit_on_plane(model, torch, origin, e1, e2, a, b):
    """The model's logit on the 2-D grid origin + a*e1 + b*e2 of its hidden space (for the boundary contour)."""
    A, Bm = np.meshgrid(a, b)
    P = origin + A.reshape(-1, 1) * e1 + Bm.reshape(-1, 1) * e2
    return head(model, P.astype(np.float32), torch).reshape(A.shape)


# =============================== assumption scores in the toy world =====================
def measures(Hr, Hc, v, n_draw):
    """Predictors of steerability computed from a vector's own pools (hidden activations Hr, Hc of the refusal /
    positive and compliance / negative pool) and the unit vector v built from a draw of n_draw per pole.
      assumption   cos(Σw v, v): 1 if v is an eigenvector of the within-class covariance (Theorem 1's condition)
      dprime       separation of the two pools along v (all pool examples)
      consistency  mean cosine of each example's deviation from the other pool's mean with v (Braun et al.)
      stability    mean cosine between v and vectors from fresh draws of the same size from the same pools"""
    v = unit(v)
    Z = np.r_[Hr - Hr.mean(0), Hc - Hc.mean(0)]
    Sv = Z.T @ (Z @ v) / (len(Z) - 2)
    pr, pc = Hr @ v, Hc @ v
    cons = np.r_[[(r - Hc.mean(0)) @ v / np.linalg.norm(r - Hc.mean(0)) for r in Hr[:2000]],
                 [(Hr.mean(0) - c) @ v / np.linalg.norm(Hr.mean(0) - c) for c in Hc[:2000]]]
    rng = np.random.default_rng(0); st = []
    for _ in range(20):
        w = Hr[rng.choice(len(Hr), min(n_draw, len(Hr)), replace=False)].mean(0) - Hc[rng.choice(len(Hc), min(n_draw, len(Hc)), replace=False)].mean(0)
        st.append(unit(w) @ v)
    return {"assumption cos(Σw v, v)": float(unit(Sv) @ v),
            "d′": float((pr.mean() - pc.mean()) / np.sqrt((pr.var() + pc.var()) / 2)),
            "consistency": float(cons.mean()), "stability": float(np.mean(st))}


def run_regime_measures(seed, regime, ft_epochs=20):
    """As run_regime, plus the predictors of `measures` for every vector (pools: the set the vector is drawn from)."""
    R = build_regime(seed, regime, ft_epochs)
    torch, rng, model, members, sets, Xe = R["torch"], R["rng"], R["model"], R["members"], R["sets"], R["Xe"]
    n_draw = min(128, int(min(members[1].sum(), (~members[1]).sum())))
    He = hidden(model, Xe, torch); nrm = boundary_normal(model, He, torch)
    scale = np.median(np.linalg.norm(He, axis=1)); doses = scale * np.linspace(0.05, 1.0, 12)
    rows = []
    for name, (X, y) in sets.items():
        a, b = class_draw(X, y, n_draw, rng)
        Hr, Hc = hidden(model, X[y], torch), hidden(model, X[~y], torch)
        v = hidden(model, a, torch).mean(0) - hidden(model, b, torch).mean(0)
        S, _ = steer_success(model, He, v, doses, torch)
        rows.append({"seed": seed, "regime": regime, "ft_epochs": ft_epochs, "vector": name, "S": S,
                     "cos to boundary normal": float(abs(unit(v) @ nrm)), **measures(Hr, Hc, v, n_draw)})
    return rows
