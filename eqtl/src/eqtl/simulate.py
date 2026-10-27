"""Expression simulation: negative binomial counts on a cis-window dosage matrix.

    y_i ~ NB(mu_i, r),   log mu_i = log l_i + log mu0 + sum_j beta_j g_ij,   r = 1 / phi,

with l_i a log-normal library-size factor relative to a reference library (log l_i ~
N(0, libsize_sd^2)), mu0 the mean count of a reference-library individual, phi the
overdispersion (DESeq2 / edgeR's alpha; phi = 0 is Poisson) and beta_j the per-allele log
fold change at the k causal sites. This is the jaxQTL pseudobulk model (log link, log
library size as offset, NB2 variance mu + phi mu^2) without the single-cell layer.

Causal sites are drawn among columns in the causal MAF band, pairwise |r| below
`causal_max_abs_r` so k causal effects are k identifiable signals. Effect signs are random.
Magnitudes follow the alpha model, |beta_j| = beta * (2 p_j (1 - p_j))^(alpha / 2) with p_j
the in-sample MAF: alpha = -1 (default) is the effect on the standardized genotype, so every
causal explains beta^2 of variance on the log scale and the expected marginal chi-square at
the causal, n beta^2 / (1 / mu0 + phi), does not depend on which site was drawn; alpha = 0 is a
frequency-independent log fold change per allele (jaxQTL's convention).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import tskit

from .genotypes import Dosage, sample_dosage

__all__ = ["ExpressionSpec", "Sim", "simulate_rep", "rep_seed", "save_sim", "load_sim", "save_batch", "load_batch"]


@dataclass(frozen=True)
class ExpressionSpec:
    n: int = 300  # individuals
    mu0: float = 100.0  # mean count at the reference library size
    phi: float = 0.2  # overdispersion 1/r; 0 = Poisson
    libsize_sd: float = 0.3  # sd of log library-size factor
    k_causal: int = 1  # 0 = null gene
    beta: float = 0.3  # |effect| at each causal, on the (2 p (1 - p))^(-alpha / 2) scaled genotype
    alpha: float = -1.0  # alpha model: |beta_j| = beta * (2 p_j (1 - p_j))^(alpha / 2)
    maf_min: float = 0.05  # design filter (in-sample MAF)
    causal_maf_min: float = 0.05
    causal_maf_max: float = 0.5
    causal_max_abs_r: float = 0.5  # pairwise |corr| cap among causal columns

    @property
    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def r(self) -> float:
        return float("inf") if self.phi == 0 else 1.0 / self.phi

    def expected_chi2(self, maf: np.ndarray | float) -> np.ndarray | float:
        """Expected marginal score chi-square at a causal of MAF `maf` (NB information for a
        log-mean effect, 1 / (1 / mu0 + phi) per individual): n beta_j^2 2 p (1 - p) * info."""
        v = 2.0 * np.asarray(maf, dtype=float) * (1.0 - np.asarray(maf, dtype=float))
        return self.n * (self.beta * v ** (self.alpha / 2.0)) ** 2 * v / (1.0 / self.mu0 + self.phi)


@dataclass(frozen=True)
class Sim:
    X: np.ndarray  # (n, p) uint8 dosage
    y: np.ndarray  # (n,) int64 counts
    log_libsize: np.ndarray  # (n,) offset
    causal_idx: np.ndarray  # (k,) column indices
    beta: np.ndarray  # (k,) signed log fold changes
    r: float  # NB size used to draw y (inf = Poisson)
    site_id: np.ndarray
    position: np.ndarray
    maf: np.ndarray
    individuals: np.ndarray

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def causal_mask(self) -> np.ndarray:
        m = np.zeros(self.p, dtype=bool)
        m[self.causal_idx] = True
        return m

    def r2_to_causal(self) -> np.ndarray:
        """Per column, the max squared correlation with any causal column (1 at a causal);
        zeros for a null gene (no causal)."""
        if self.causal_idx.shape[0] == 0:
            return np.zeros(self.p)
        Xc = self.X.astype(float) - self.X.mean(axis=0)
        sd = np.sqrt((Xc**2).sum(axis=0))
        sd[sd == 0] = np.inf
        C = Xc[:, self.causal_idx]
        r = (Xc.T @ C) / np.outer(sd, sd[self.causal_idx])
        return (r**2).max(axis=1)


def rep_seed(cell_hash: str, rep: int) -> int:
    h = hashlib.sha256(f"{cell_hash}:{rep}".encode()).hexdigest()
    return int(h[:8], 16)


def _draw_causal(X: np.ndarray, maf: np.ndarray, spec: ExpressionSpec, rng: np.random.Generator) -> np.ndarray:
    if spec.k_causal == 0:
        return np.zeros(0, dtype=np.int64)
    eligible = np.flatnonzero((maf >= spec.causal_maf_min) & (maf <= spec.causal_maf_max))
    if eligible.shape[0] < spec.k_causal:
        raise ValueError(f"only {eligible.shape[0]} columns in the causal MAF band for k={spec.k_causal}")
    order = rng.permutation(eligible)
    Xf = X.astype(float)
    chosen: list[int] = []
    for j in order:
        ok = True
        for c in chosen:
            r = np.corrcoef(Xf[:, j], Xf[:, c])[0, 1]
            if abs(r) > spec.causal_max_abs_r:
                ok = False
                break
        if ok:
            chosen.append(int(j))
        if len(chosen) == spec.k_causal:
            break
    if len(chosen) < spec.k_causal:
        raise ValueError(f"could not place {spec.k_causal} causal sites with pairwise |r| <= {spec.causal_max_abs_r}")
    return np.array(sorted(chosen), dtype=np.int64)


def _draw_counts(mu: np.ndarray, r: float, rng: np.random.Generator) -> np.ndarray:
    if not np.isfinite(r):
        return rng.poisson(mu).astype(np.int64)
    # numpy's NB(n, p) has mean n (1-p)/p, so p = r / (r + mu)
    return rng.negative_binomial(r, r / (r + mu)).astype(np.int64)


def simulate_rep(ts: tskit.TreeSequence, spec: ExpressionSpec, seed: int) -> Sim:
    rng = np.random.default_rng(seed)
    d: Dosage = sample_dosage(ts, spec.n, rng, maf_min=spec.maf_min)
    causal = _draw_causal(d.X, d.maf, spec, rng)
    signs = rng.choice([-1.0, 1.0], size=causal.shape[0])
    v = 2.0 * d.maf[causal] * (1.0 - d.maf[causal])
    beta = signs * spec.beta * v ** (spec.alpha / 2.0)
    log_libsize = rng.normal(0.0, spec.libsize_sd, size=spec.n)
    eta = log_libsize + np.log(spec.mu0) + d.X[:, causal].astype(float) @ beta
    y = _draw_counts(np.exp(eta), spec.r, rng)
    return Sim(
        X=d.X, y=y, log_libsize=log_libsize, causal_idx=causal, beta=beta, r=spec.r,
        site_id=d.site_id, position=d.position, maf=d.maf, individuals=d.individuals,
    )


def save_sim(sim: Sim, path: str) -> None:
    np.savez_compressed(
        path, X=sim.X, y=sim.y, log_libsize=sim.log_libsize, causal_idx=sim.causal_idx,
        beta=sim.beta, r=np.array(sim.r), site_id=sim.site_id, position=sim.position,
        maf=sim.maf, individuals=sim.individuals,
    )


def load_sim(path: str) -> Sim:
    with np.load(path) as z:
        return Sim(
            X=z["X"], y=z["y"], log_libsize=z["log_libsize"], causal_idx=z["causal_idx"],
            beta=z["beta"], r=float(z["r"]), site_id=z["site_id"], position=z["position"],
            maf=z["maf"], individuals=z["individuals"],
        )


_SIM_FIELDS = ("X", "y", "log_libsize", "causal_idx", "beta", "r", "site_id", "position", "maf", "individuals")


def save_batch(sims: dict[int, Sim], path: str) -> None:
    """One file per replicate batch: every array of every rep, keyed `<rep>/<field>`."""
    arrays = {}
    for rep, sim in sims.items():
        for f in _SIM_FIELDS:
            v = getattr(sim, f)
            arrays[f"{rep}/{f}"] = np.array(v) if f == "r" else v
    np.savez_compressed(path, **arrays)


def load_batch(path: str) -> dict[int, Sim]:
    """{rep: Sim} from `save_batch`, in replicate order."""
    out = {}
    with np.load(path) as z:
        reps = sorted({int(k.split("/")[0]) for k in z.files})
        for rep in reps:
            g = {f: z[f"{rep}/{f}"] for f in _SIM_FIELDS}
            g["r"] = float(g["r"])
            out[rep] = Sim(**g)
    return out
