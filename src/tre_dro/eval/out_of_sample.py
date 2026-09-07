"""Out-of-sample evaluation with exact capacity-constrained allocation.

Repaired from ``run_supplementary_experiments.py:415-482`` which had the
correct lognormal parameterisation; the other 3 files had
    ln_sigma = log(sigma²+mu²)  (wrong)
The correct form is:
    ln_sigma = sqrt(log(1 + CV²))

The multivariate mode uses a Gaussian copula built from a correlation matrix;
the lognormal marginal parameters preserve the supplied arithmetic means and
standard deviations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.schema import Instance


@dataclass
class OOSResult:
    method: str
    mean_cost: float
    std_cost: float
    cv_cost: float
    p95: float
    p99: float
    cvar_95: float
    mean_unmet_ton: float
    mean_capacity_utilization: float
    n_scenarios: int
    seed: int
    extra: dict[str, Any] = field(default_factory=dict)


class OutOfSampleEvaluator:
    """Evaluate a fixed facility selection over many demand scenarios."""

    def __init__(
        self,
        n_scenarios: int = 2000,
        seed: int = 999,
        alpha_cvar: float = 0.95,
        unmet_penalty: float | None = None,
        demand_model: str = "lognormal",  # or "multivariate_lognormal"
        correlation_matrix: np.ndarray | None = None,
    ):
        self.n_scenarios = n_scenarios
        self.seed = seed
        self.alpha_cvar = alpha_cvar
        self.unmet_penalty = unmet_penalty
        self.demand_model = demand_model
        self.correlation_matrix = correlation_matrix

    def evaluate(
        self,
        instance: Instance,
        y_open: np.ndarray,
        method_name: str,
    ) -> OOSResult:
        rng = np.random.RandomState(self.seed)
        n, m = instance.distance_km.shape
        fixed_cost = float(np.sum(instance.fixed_costs * y_open))
        costs_per_km = (
            instance.cost_per_ton_km * instance.distance_km
            + instance.processing_costs[None, :]
        )

        if self.demand_model == "lognormal":
            scenarios = _lognormal_scenarios(instance, self.n_scenarios, rng)
        elif self.demand_model == "multivariate_lognormal":
            scenarios = _multivariate_lognormal_scenarios(
                instance, self.n_scenarios, rng,
                correlation_matrix=self.correlation_matrix,
            )
        else:
            raise ValueError(self.demand_model)

        per_scenario = np.zeros(self.n_scenarios)
        unmet_per_scenario = np.zeros(self.n_scenarios)
        cap_util = np.zeros(self.n_scenarios)

        penalty = instance.unmet_penalty if self.unmet_penalty is None else self.unmet_penalty
        open_idx = np.where(np.asarray(y_open, float) > 0.5)[0]
        transport_cost = costs_per_km[:, open_idx]
        n_open = len(open_idx)
        n_flow = n * n_open
        c_obj = np.concatenate([transport_cost.reshape(-1), np.full(n, penalty)])
        bounds = [(0.0, None)] * (n_flow + n)

        A_eq = np.zeros((n, n_flow + n))
        for i in range(n):
            A_eq[i, i * n_open:(i + 1) * n_open] = 1.0
            A_eq[i, n_flow + i] = 1.0
        A_ub = np.zeros((n_open, n_flow + n)) if n_open else None
        if n_open:
            for jj in range(n_open):
                A_ub[jj, jj:n_flow:n_open] = 1.0
        b_ub = instance.capacities[open_idx] if n_open else None

        from scipy.optimize import linprog

        for k in range(self.n_scenarios):
            d = scenarios[k]
            lp = linprog(
                c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=d,
                bounds=bounds, method="highs",
            )
            if not lp.success:
                raise RuntimeError(f"OOS recourse LP failed for scenario {k}: {lp.message}")
            unmet = float(np.sum(lp.x[n_flow:]))
            used_capacity = float(np.sum(lp.x[:n_flow]))
            per_scenario[k] = fixed_cost + float(lp.fun)
            unmet_per_scenario[k] = unmet
            total_open_capacity = float(np.sum(instance.capacities[open_idx]))
            cap_util[k] = used_capacity / max(1e-9, total_open_capacity)

        mean_c = float(np.mean(per_scenario))
        std_c = float(np.std(per_scenario, ddof=1))
        sorted_c = np.sort(per_scenario)
        p95_v = float(sorted_c[int(np.ceil(self.alpha_cvar * self.n_scenarios)) - 1])
        tail = sorted_c[int(np.ceil(self.alpha_cvar * self.n_scenarios)) - 1:]
        cvar = float(np.mean(per_scenario[per_scenario >= p95_v])) if len(tail) < self.n_scenarios else mean_c
        return OOSResult(
            method=method_name,
            mean_cost=mean_c, std_cost=std_c,
            cv_cost=std_c / max(1.0, mean_c),
            p95=p95_v, p99=float(sorted_c[int(0.99 * self.n_scenarios) - 1]),
            cvar_95=cvar,
            mean_unmet_ton=float(np.mean(unmet_per_scenario)),
            mean_capacity_utilization=float(np.mean(cap_util)),
            n_scenarios=self.n_scenarios, seed=self.seed,
            extra={
                "scenario_costs": per_scenario,
                "scenario_unmet_ton": unmet_per_scenario,
                "demand_model": self.demand_model,
                "scenario_mean": np.mean(scenarios, axis=0),
                "scenario_std": np.std(scenarios, axis=0, ddof=1),
                "scenario_correlation": np.corrcoef(scenarios, rowvar=False),
            },
        )


def _lognormal_scenarios(instance: Instance, n_scenarios: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate n_scenarios × n univariate lognormal demands.

    Correct parametrisation:
        ln_sigma = sqrt(log(1 + CV^2))
        ln_mu = log(mu) - 0.5 * ln_sigma^2
    """
    n = instance.n_customers
    mu = instance.mu.astype(float)
    sigma = instance.sigma.astype(float)
    cv = np.where(mu > 0, sigma / mu, 0.0)
    ln_sigma = np.sqrt(np.log(1.0 + cv ** 2))
    ln_mu = np.log(np.maximum(mu, 1e-6)) - 0.5 * ln_sigma ** 2
    out = np.zeros((n_scenarios, n))
    for i in range(n):
        out[:, i] = rng.lognormal(ln_mu[i], ln_sigma[i], size=n_scenarios)
    return out


def _multivariate_lognormal_scenarios(
    instance: Instance, n_scenarios: int, rng: np.random.RandomState,
    correlation_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Generate multivariate lognormal demand with a Gaussian copula.

    Uses the covariance Σ from the ambiguity set (if available in instance
    provenance). Falls back to independent lognormal.
    """
    n = instance.n_customers
    sigma_matrix = correlation_matrix
    if sigma_matrix is None:
        sigma_matrix = instance.provenance.get("sigma_matrix")
    if sigma_matrix is not None:
        from ..ambiguity.covariance import cholesky_sqrt as chol
        covariance = np.asarray(sigma_matrix, float)
        scale = np.sqrt(np.maximum(np.diag(covariance), 1e-30))
        corr = covariance / (scale[:, None] * scale[None, :])
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)
        mu = instance.mu.astype(float)
        cv = instance.sigma / np.maximum(mu, 1e-6)
        ln_sigma = np.sqrt(np.log(1.0 + cv ** 2))
        denom = ln_sigma[:, None] * ln_sigma[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            latent_corr = np.where(
                denom > 0,
                np.log1p(corr * cv[:, None] * cv[None, :]) / denom,
                0.0,
            )
        latent_corr = np.clip(latent_corr, -0.999999, 0.999999)
        np.fill_diagonal(latent_corr, 1.0)
        # Numerical projection may be needed because pairwise moment matching
        # does not guarantee a PSD latent matrix in finite samples.
        from ..ambiguity.covariance import _psd_floor
        latent_corr = _psd_floor(latent_corr, 1e-12)
        latent_scale = np.sqrt(np.maximum(np.diag(latent_corr), 1e-30))
        latent_corr /= latent_scale[:, None] * latent_scale[None, :]
        np.fill_diagonal(latent_corr, 1.0)
        L = chol(latent_corr)
        z = rng.randn(n_scenarios, n) @ L.T
        ln_mu = np.log(np.maximum(mu, 1e-6)) - 0.5 * ln_sigma ** 2
        from scipy import stats
        uniforms = np.clip(stats.norm.cdf(z), 1e-12, 1.0 - 1e-12)
        out = np.zeros((n_scenarios, n))
        for i in range(n):
            out[:, i] = stats.lognorm.ppf(
                uniforms[:, i], s=ln_sigma[i], scale=np.exp(ln_mu[i]))
        return out
    else:
        return _lognormal_scenarios(instance, n_scenarios, rng)
