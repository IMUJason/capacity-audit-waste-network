"""Statistical tests: paired t, Wilcoxon, Cohen's d, Holm correction.

Ported from ``code/_archive_unused_scripts/statistical_validation.py:215-364``.
Paired versions added (not just Welch); Bootstrap CI in bootstrap.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PairedTestResult:
    method_a: str
    method_b: str
    t_statistic: float | None
    p_value: float | None
    wilcoxon_p: float
    cohens_d: float | None
    ci_lower: float
    ci_upper: float
    n_samples: int
    mean_diff: float
    std_diff: float
    degenerate: bool
    note: str | None


def paired_test(
    a: np.ndarray, b: np.ndarray, method_a: str = "A", method_b: str = "B",
    confidence: float = 0.95,
) -> PairedTestResult:
    """Paired-sample t-test + Wilcoxon + Cohen's d + CI."""
    a_f = np.asarray(a, float)
    b_f = np.asarray(b, float)
    if a_f.shape != b_f.shape or a_f.ndim != 1 or len(a_f) < 2:
        raise ValueError("paired samples must be one-dimensional, equal-length, and n >= 2")
    diff = a_f - b_f
    n = len(diff)
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    zero_variance = bool(
        np.isclose(std_d, 0.0, atol=1e-12 * max(1.0, abs(mean_d)), rtol=0.0)
    )
    if np.allclose(diff, 0.0):
        t_statistic, p_value, wilcoxon_p = 0.0, 1.0, 1.0
        d: float | None = 0.0
        note = "The paired outcomes are identical; inferential tests are unnecessary."
    elif zero_variance:
        # A constant non-zero paired difference has no sampling variance. A
        # t-statistic and standardized effect size are undefined, not infinite.
        t_statistic, p_value = None, None
        wilcoxon_p = float(stats.wilcoxon(a_f, b_f).pvalue)
        d = None
        note = (
            "All paired differences are the same non-zero value; the t-test "
            "and Cohen's d are undefined. Report the deterministic difference."
        )
    else:
        t_res = stats.ttest_rel(a_f, b_f)
        w_res = stats.wilcoxon(a_f, b_f)
        t_statistic = float(t_res.statistic)
        p_value = float(t_res.pvalue)
        wilcoxon_p = float(w_res.pvalue)
        d = cohens_d_from_diff(diff)
        note = None
    se = std_d / np.sqrt(n) if n > 1 else 1.0
    t_crit = stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=n - 1) if n > 1 else 1.96
    return PairedTestResult(
        method_a=method_a, method_b=method_b,
        t_statistic=t_statistic, p_value=p_value,
        wilcoxon_p=wilcoxon_p,
        cohens_d=d, ci_lower=mean_d - t_crit * se, ci_upper=mean_d + t_crit * se,
        n_samples=n, mean_diff=mean_d, std_diff=std_d,
        degenerate=zero_variance, note=note,
    )


def cohens_d_from_diff(diff: np.ndarray) -> float:
    d = np.asarray(diff, float)
    return float(np.mean(d) / max(np.std(d, ddof=1), 1e-9))


def holm_corrected_pvalues(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for a list of p-values."""
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.zeros(n, float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(float(p_values[idx]) * (n - rank), 1.0)
        running_max = max(running_max, adjusted)
        corrected[idx] = running_max
    return corrected.tolist()
