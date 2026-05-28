"""
Evaluation Metrics
==================
Implements all five primary KPIs reported in the paper (Section 5.6):
  1. Makespan  (seconds)               — lower is better
  2. SLA Violation Rate (%)            — lower is better
  3. Deadline Miss Rate, DMR (%)       — lower is better
  4. CPU Utilization (%)               — higher is better
  5. Energy Consumption (kWh)          — lower is better

Also implements:
  - One-way ANOVA with post-hoc Tukey HSD (Section 6.6)
  - Cohen's d effect size (Section 6.6)
  - LSTM-AE evaluation: RMSE, Precision, Recall, F1 (Section 6.7)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


# ---------------------------------------------------------------------------
# Scheduling KPIs
# ---------------------------------------------------------------------------


def compute_makespan(finish_times: np.ndarray) -> float:
    """
    Makespan = max completion time across all tasks.
    Equation: Makespan = max{i,j}{ T_ij · x_ij }

    Args:
        finish_times: (n_tasks,) array of task finish timestamps (seconds)
    Returns:
        makespan in seconds
    """
    return float(np.max(finish_times))


def compute_sla_violation_rate(
    finish_times: np.ndarray,
    sla_deadlines: np.ndarray,
) -> float:
    """
    SLA Violation Rate = % tasks breaching negotiated SLA deadlines.

    Args:
        finish_times:  (n_tasks,) actual finish timestamps
        sla_deadlines: (n_tasks,) SLA deadline timestamps
    Returns:
        violation rate in [0, 100]
    """
    violations = np.sum(finish_times > sla_deadlines)
    return float(100.0 * violations / len(finish_times))


def compute_deadline_miss_rate(
    finish_times: np.ndarray,
    task_deadlines: np.ndarray,
) -> float:
    """
    Deadline Miss Rate (DMR) = % tasks completing after their absolute deadline.
    DMR = |{i : T_ij · x_ij > D_i}| / n  (Equation in Section 3.3)

    Note: SLA deadlines and task deadlines may differ — SLA thresholds are
    typically looser per-contract bounds, while task deadlines are hard limits.

    Args:
        finish_times:    (n_tasks,)
        task_deadlines:  (n_tasks,)
    Returns:
        DMR in [0, 100]
    """
    misses = np.sum(finish_times > task_deadlines)
    return float(100.0 * misses / len(finish_times))


def compute_cpu_utilization(
    assigned_compute: np.ndarray,    # C_i for assigned tasks
    resource_capacities: np.ndarray, # M_j for each resource
    epoch_duration: float,
) -> float:
    """
    CPU Utilization = (Σ C_i · x_ij) / (Σ M_j · T_epoch)

    Args:
        assigned_compute:   (n_tasks,) compute demand in MIPS-seconds
        resource_capacities: (n_resources,) processing capacity in MIPS
        epoch_duration:     epoch length in seconds
    Returns:
        CPU utilization in [0, 100]
    """
    total_assigned = float(np.sum(assigned_compute))
    total_capacity = float(np.sum(resource_capacities)) * epoch_duration
    return float(100.0 * total_assigned / (total_capacity + 1e-8))


def compute_energy_consumption(
    energy_per_task: np.ndarray,
) -> float:
    """
    Total energy = Σ_i Σ_j E_ij · x_ij  (Equation 2 in the paper)

    Args:
        energy_per_task: (n_tasks,) energy in kWh per task execution
    Returns:
        total energy in kWh
    """
    return float(np.sum(energy_per_task))


def compute_all_kpis(
    finish_times: np.ndarray,
    sla_deadlines: np.ndarray,
    task_deadlines: np.ndarray,
    assigned_compute: np.ndarray,
    resource_capacities: np.ndarray,
    energy_per_task: np.ndarray,
    epoch_duration: float,
) -> Dict[str, float]:
    """
    Convenience wrapper: compute all five KPIs in one call.
    Returns a dict matching the columns in Table 4 of the paper.
    """
    return {
        "makespan_s": compute_makespan(finish_times),
        "sla_violation_pct": compute_sla_violation_rate(finish_times, sla_deadlines),
        "dmr_pct": compute_deadline_miss_rate(finish_times, task_deadlines),
        "cpu_util_pct": compute_cpu_utilization(
            assigned_compute, resource_capacities, epoch_duration
        ),
        "energy_kwh": compute_energy_consumption(energy_per_task),
    }


# ---------------------------------------------------------------------------
# Statistical Tests (Section 6.6)
# ---------------------------------------------------------------------------


def one_way_anova(*groups: np.ndarray) -> Tuple[float, float]:
    """
    One-way ANOVA across scheduler result groups.
    Returns:
        F-statistic, p-value
    """
    result = stats.f_oneway(*groups)
    return float(result.statistic), float(result.pvalue)


def eta_squared(f_stat: float, n_groups: int, total_n: int) -> float:
    """
    Eta-squared (η²) effect size for ANOVA.
    η² = SS_between / SS_total ≈ (F·df_between) / (F·df_between + df_within)

    Args:
        f_stat:   F-statistic from ANOVA
        n_groups: number of groups (schedulers)
        total_n:  total number of observations
    Returns:
        η² in [0, 1]
    """
    df_between = n_groups - 1
    df_within = total_n - n_groups
    return float((f_stat * df_between) / (f_stat * df_between + df_within))


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Cohen's d effect size: (μ_a - μ_b) / pooled_std

    Interpretation:
        |d| < 0.2  → negligible
        |d| < 0.5  → small
        |d| < 0.8  → medium
        |d| < 1.2  → large
        |d| >= 1.2 → very large

    Args:
        group_a, group_b: 1-D arrays of metric values
    Returns:
        Cohen's d (signed; positive means group_a > group_b)
    """
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    return float((np.mean(group_a) - np.mean(group_b)) / (pooled_std + 1e-12))


def tukey_hsd_pairwise(groups: Dict[str, np.ndarray]) -> List[dict]:
    """
    Post-hoc Tukey HSD pairwise comparisons (requires pingouin or statsmodels).
    Reproduces Table 7 in the paper.

    Args:
        groups: dict {scheduler_name: array_of_metric_values}
    Returns:
        list of dicts with keys: pair, mean_diff, p_value, cohens_d, significant
    """
    try:
        import pingouin as pg
        import pandas as pd

        records = []
        for name, vals in groups.items():
            for v in vals:
                records.append({"scheduler": name, "value": v})
        df = pd.DataFrame(records)
        result = pg.pairwise_tukey(data=df, dv="value", between="scheduler")
        return result.to_dict(orient="records")
    except ImportError:
        # Fallback: manual pairwise t-tests (Bonferroni-corrected)
        names = list(groups.keys())
        results = []
        n_pairs = len(names) * (len(names) - 1) // 2
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = groups[names[i]], groups[names[j]]
                t_stat, p_val = stats.ttest_ind(a, b)
                p_bonf = min(1.0, p_val * n_pairs)
                d = cohens_d(a, b)
                results.append({
                    "A": names[i],
                    "B": names[j],
                    "mean_diff": float(np.mean(a) - np.mean(b)),
                    "p_value_bonferroni": float(p_bonf),
                    "cohens_d": d,
                    "significant": p_bonf < 0.05,
                })
        return results


# ---------------------------------------------------------------------------
# LSTM-AE Metrics (Section 6.7)
# ---------------------------------------------------------------------------


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error as percentage of range."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    value_range = y_true.max() - y_true.min() + 1e-8
    return float(100.0 * rmse / value_range)


def compute_anomaly_metrics(
    y_true_binary: np.ndarray, y_pred_binary: np.ndarray
) -> Dict[str, float]:
    """
    Precision, Recall, F1 for binary anomaly detection.
    Paper reports: precision=0.96, recall=0.92, F1=0.94 (Section 6.7).
    """
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))
    return {"precision": precision, "recall": recall, "f1": f1}
