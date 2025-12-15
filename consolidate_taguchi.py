#!/usr/bin/env python3
"""
consolidate_taguchi_plus.py - Publication-Ready Statistical Analysis

Enhanced consolidation script with rigorous statistical testing:
- Single-sided t-tests comparing runs against C3D baseline
- FDR correction (Benjamini-Hochberg) for multiple comparisons
- One-sample Cohen's d effect sizes
- 95% confidence intervals (t-based or bootstrap)
- Confusion matrix reconstruction with validation
- Organized output structure for publication
- Minimal, LaTeX-compatible titles

Key Features:
1. Statistical rigor: FDR-corrected p-values, effect sizes, confidence intervals
2. Reproducibility: Seed control, validation checks, provenance tracking
3. Publication-ready: 300 DPI figures (PNG + PDF), clean formatting
4. Flexibility: CLI-overridable baseline, bootstrap option, custom alpha

Confusion Matrix Outputs:
- Per-fold: Individual fold CMs with row-normalized percentages
- Aggregated (micro): Summed across folds, then row-normalized
- Macro average: Mean of per-fold row-normalized CMs (no fold dominates)

Statistical Testing Notes:
The script performs two types of statistical comparisons:

1. **All Runs Comparison (FDR-corrected)**:
   - Compares each run against single C3D baseline value
   - Tests multiple metrics × runs → requires FDR correction
   - Appropriate for exploratory analysis across all experimental conditions

2. **Best Run Analysis (per-metric)**:
   - Identifies best-performing run for each key metric:
     * Risky Recall, Precision, F1
     * Accuracy
     * Safe Recall
   - Tests if best run significantly exceeds baseline (single-sided t-test)
   - No FDR correction (pre-selected best runs)
   - Use for "our best configuration achieved X" claims

Important Limitations:
- Tests use single baseline value (no baseline variance) → conservative inference
- For definitive "statistically superior to C3D" claims, need:
  1. C3D metrics on same folds (paired data)
  2. Paired t-test or Wilcoxon signed-rank test
  3. Pre-declared primary metric to avoid multiplicity

Without paired data: Report descriptive improvements with CIs and avoid strong
significance claims. Current tests answer "does our best run exceed baseline?"
not "is our method statistically superior?"

Usage Examples:
  # Basic usage
  python consolidate_taguchi_plus.py --results_dir /path/to/GRADCAM_RESULTS
  
  # With custom baseline and bootstrap CI
  python consolidate_taguchi_plus.py \\
      --results_dir /path/to/results \\
      --baseline_tp 10 --baseline_fp 5 --baseline_tn 25 --baseline_fn 8 \\
      --use_bootstrap --n_bootstrap 10000 --seed 42
  
  # Strict significance threshold
  python consolidate_taguchi_plus.py \\
      --results_dir /path/to/results \\
      --alpha 0.01

Output Structure:
  CONSOLIDATED_RESULTS/
  ├── Tables/
  │   ├── Statistical/
  │   │   ├── statistical_tests.csv          # FDR-corrected tests (all runs)
  │   │   └── best_run_tests.csv             # Best run per metric vs baseline
  │   ├── summary_statistics.csv             # Macro means + CI
  │   ├── cm_components_micro.csv            # Micro metrics from summed CMs
  │   └── cm_reconstruction_validation.csv   # Reconstruction quality checks
  ├── ConfusionMatrices/
  │   ├── Baseline/                          # C3D baseline CM
  │   ├── Optimal/PerFold/                   # Individual fold CMs
  │   ├── Optimal/Summed/
  │   │   ├── *_summed.png                   # Micro (pooled across folds)
  │   │   └── *_macroAvg.png                 # Macro (avg of fold-normalized)
  │   ├── Balanced/PerFold/
  │   ├── Balanced/Summed/
  │   └── Best/                              # Best CMs by metric
  ├── Plots/Optimal/                         # Performance visualizations
  ├── Plots/Balanced/
  ├── Curves/Optimal/                        # PR/ROC curves
  ├── Curves/Balanced/
  └── STATISTICAL_SUMMARY.txt                # Human-readable summary

Statistical Notes:
- Single-sided tests: H0: run ≤ baseline, H1: run > baseline
- FDR correction controls false discovery rate across all comparisons
- Cohen's d (one-sample): (mean - baseline) / SD
- Bootstrap CI available for robust non-parametric intervals
- Reconstruction tolerance: ±0.5% for precision/recall validation
"""

import re
import shutil
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Statsmodels with fallback for FDR correction
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[warn] statsmodels not available, using fallback FDR implementation")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Publication-quality figure settings
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ----------------------------- CONFIG ---------------------------------

#DEFAULT_RESULTS_DIR = "/ocean/projects/asc180003p/szaidi/Tackle_Ablation/taguchi_runs_GRADCAM_RESULTS/"

DEFAULT_RESULTS_DIR ="/ocean/projects/asc180003p/szaidi/Tackle_Ablation/Results_v3_a_55-y_1.3/"

RUN_RENAMES = {
    "run_0_original": "No_augmentation",
    "run_0": "Oversampled_NoAugmentation",
}
BASELINE_NAME = "C3D Baseline"
# CLI-overridable baseline confusion matrix
BASELINE_CM = dict(TP=7, FP=6, TN=20, FN=5)

# Reconstruction tolerance for validation (±0.5% for precision/recall validation)
CM_RECONSTRUCTION_TOLERANCE = 0.005

def _baseline_counts_lower() -> Dict[str, float]:
    """Convert BASELINE_CM uppercase keys to lowercase for metrics_from_counts()"""
    return {
        "tp": BASELINE_CM["TP"],
        "fp": BASELINE_CM["FP"],
        "tn": BASELINE_CM["TN"],
        "fn": BASELINE_CM["FN"],
    }

# Column families
PRIMARY_METRICS = [
    "opt_accuracy", "opt_risky_precision", "opt_risky_recall", "opt_risky_f1",
    "opt_safe_recall", "opt_safe_f1", "optimal_threshold"
]
SECONDARY_METRICS = [
    "std_accuracy", "std_risky_precision", "std_risky_recall", "std_risky_f1",
    "std_safe_recall", "std_safe_f1"
]
CURVE_METRICS = ["risky_ap", "safe_ap", "risky_roc_auc", "safe_roc_auc"]
METRICS_OF_INTEREST = PRIMARY_METRICS + SECONDARY_METRICS + CURVE_METRICS

# For optional PR/ROC CSV point discovery
PR_GLOBS = ["*pr*curve*safe*.csv", "*pr*safe*.csv", "*precision*recall*safe*.csv",
            "*pr*curve*risky*.csv", "*pr*risky*.csv", "*precision*recall*risky*.csv"]
ROC_GLOBS = ["*roc*curve*safe*.csv", "*roc*safe*.csv", "*fpr*tpr*safe*.csv",
             "*roc*curve*risky*.csv", "*roc*risky*.csv", "*fpr*tpr*risky*.csv"]

# Regex helpers to discover run/fold
_RUN_PAT = re.compile(r"(?:^|[^a-z])run[-_]?(\d+)(?:[^a-z]|$)", re.I)
_FOLD_PAT = re.compile(r"(?:^|[^a-z])fold[-_]?(\d+)(?:[^a-z]|$)", re.I)
_COMBINED_PAT = re.compile(r"run[-_]?(\d+)[-_]?fold[-_]?(\d+)", re.I)

# Regex patterns for parsing validation set counts
_VAL_RISKY_PAT = re.compile(r"Risky\s*\(label\s*=\s*1\)\s*:\s*(\d+)", re.I)
_VAL_SAFE_PAT  = re.compile(r"Safe\s*\(label\s*=\s*0\)\s*:\s*(\d+)", re.I)

# ------------------------- SMALL UTILS ---------------------------------

def rename_run(run: str) -> str:
    return RUN_RENAMES.get(run, run)

def parse_run_fold_from_path(p: Path) -> Optional[Tuple[str, int]]:
    parts = [x.name for x in p.parents]
    for s in parts:
        m = _COMBINED_PAT.search(s)
        if m:
            return (f"run_{int(m.group(1))}", int(m.group(2)))
    run_num = None; fold_num = None
    for s in parts:
        r = _RUN_PAT.search(s)
        f = _FOLD_PAT.search(s)
        if r and run_num is None: run_num = int(r.group(1))
        if f and fold_num is None: fold_num = int(f.group(1))
    if run_num is not None and fold_num is not None:
        return (f"run_{run_num}", fold_num)
    return None

def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        if df.empty: return None
        return df
    except Exception:
        return None

def ensure_dirs(base_out: Path):
    for sub in [
        "ConfusionMatrices/Optimal/PerFold", 
        "ConfusionMatrices/Optimal/Summed",
        "ConfusionMatrices/Balanced/PerFold", 
        "ConfusionMatrices/Balanced/Summed",
        "ConfusionMatrices/Best",
        "ConfusionMatrices/Baseline",
        "Plots/Optimal", "Plots/Balanced", 
        "Tables/Statistical", 
        "Curves/Optimal", "Curves/Balanced"
    ]:
        (base_out / sub).mkdir(parents=True, exist_ok=True)

# -------------------- STATISTICAL TESTING ------------------------------

def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Uses statsmodels if available, otherwise falls back to manual implementation.
    
    Args:
        p_values: Array of p-values
        alpha: Family-wise error rate
    
    Returns:
        (q_values, significant) tuple where:
        - q_values: FDR-corrected q-values
        - significant: Boolean array indicating significance at alpha level
    """
    p_values = np.asarray(p_values, dtype=float)
    
    if HAS_STATSMODELS:
        _, q_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        significant = q_values < alpha
    else:
        # Fallback manual Benjamini-Hochberg implementation
        n = len(p_values)
        if n == 0:
            return np.array([]), np.array([], dtype=bool)
        
        # Sort p-values and track original order
        order = np.argsort(p_values)
        ranked_p = p_values[order]
        
        # Calculate q-values: p * n / rank
        ranks = np.arange(1, n + 1)
        q_ranked = ranked_p * n / ranks
        
        # Enforce monotonicity (q-values should be non-decreasing)
        q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
        
        # Restore original order
        q_values = np.empty_like(q_ranked)
        q_values[order] = np.minimum(q_ranked, 1.0)
        
        significant = q_values < alpha
    
    return q_values, significant

def cohens_d_one_sample(data: np.ndarray, baseline_value: float) -> float:
    """
    Calculate one-sample Cohen's d effect size.
    d = (sample_mean - baseline) / sample_std
    """
    if len(data) == 0:
        return np.nan
    mean_diff = np.mean(data) - baseline_value
    std = np.std(data, ddof=1)
    return mean_diff / std if std > 0 else np.nan

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, alpha: float = 0.05, 
                 seed: Optional[int] = None) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for the mean using local RNG.
    
    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap resamples
        alpha: Significance level (default 0.05 for 95% CI)
        seed: Random seed for reproducibility (uses local Generator, not global state)
    
    Returns:
        (lower_bound, upper_bound) tuple
    """
    # Use local random number generator to avoid global state corruption
    rng = np.random.default_rng(seed)
    
    n = len(data)
    # Vectorized bootstrap: sample indices with replacement
    bootstrap_samples = rng.choice(data, size=(n_bootstrap, n), replace=True)
    bootstrap_means = bootstrap_samples.mean(axis=1)
    
    lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    return lower, upper

def statistical_comparison(results_df: pd.DataFrame, baseline_metrics: Dict[str, float], 
                          alpha: float = 0.05, use_bootstrap: bool = False,
                          n_bootstrap: int = 10000, seed: Optional[int] = None,
                          per_threshold_fdr: bool = True) -> pd.DataFrame:
    """
    Perform single-sided t-test comparing each run against baseline.
    H0: run_metric <= baseline_metric
    H1: run_metric > baseline_metric (we want improvement)
    
    Args:
        results_df: DataFrame with per-fold metrics
        baseline_metrics: Dict with baseline values (keys: 'accuracy', 'precision', 'recall', 'f1')
        alpha: Significance level
        use_bootstrap: If True, use bootstrap CI instead of t-based CI
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed for reproducibility
        per_threshold_fdr: If True, compute FDR separately for optimal/balanced families
    
    Returns:
        DataFrame with statistical test results including FDR-corrected p-values
    """
    statistical_results = []
    
    for run in results_df['run'].unique():
        run_data = results_df[results_df['run'] == run]
        run_display = rename_run(run)
        
        for threshold_type in ['optimal', 'balanced']:
            prefix = 'opt_' if threshold_type == 'optimal' else 'std_'
            
            # Map metric names: metrics_from_counts returns generic names,
            # but CSV columns have risky_* prefix
            metric_mapping = {
                'accuracy': 'accuracy',
                'precision': 'risky_precision',
                'recall': 'risky_recall', 
                'f1': 'risky_f1'
            }
            
            for metric_base, metric_col_suffix in metric_mapping.items():
                metric_col = f'{prefix}{metric_col_suffix}'
                
                if metric_col not in run_data.columns:
                    continue
                
                # Get run data for this metric
                run_values = run_data[metric_col].dropna().values
                
                if len(run_values) == 0:
                    continue
                
                # Baseline value - use mapped key
                baseline_value = baseline_metrics.get(metric_base, np.nan)
                
                if not np.isfinite(baseline_value):
                    continue
                
                # Calculate statistics
                run_mean = np.mean(run_values)
                run_std = np.std(run_values, ddof=1)
                n_folds = len(run_values)
                
                # Confidence interval: t-based or bootstrap
                if use_bootstrap and n_folds >= 3:
                    ci_low, ci_high = bootstrap_ci(run_values, n_bootstrap, alpha, seed)
                    ci_method = 'bootstrap'
                elif n_folds > 1:
                    # Standard t-based CI
                    ci_low, ci_high = scipy_stats.t.interval(
                        1 - alpha,
                        n_folds - 1,
                        loc=run_mean,
                        scale=scipy_stats.sem(run_values)
                    )
                    ci_method = 't-distribution'
                else:
                    ci_low = ci_high = run_mean
                    ci_method = 'single_value'
                
                # Single-sided t-test
                # H0: mean <= baseline, H1: mean > baseline (one-sided, upper tail)
                if n_folds > 1:
                    try:
                        # Use built-in one-sided test if available (scipy >= 1.6.0)
                        res = scipy_stats.ttest_1samp(run_values, baseline_value, alternative='greater')
                        t_statistic = res.statistic
                        p_value = res.pvalue
                    except TypeError:
                        # Fallback for older scipy versions
                        t_statistic, p_value_two_sided = scipy_stats.ttest_1samp(run_values, baseline_value)
                        # Convert to one-sided: we only care if run > baseline
                        # If t > 0, the run mean exceeds baseline, use p/2 (upper tail)
                        # If t <= 0, the run mean is below baseline, use 1 - p/2 (not significant in our direction)
                        p_value = p_value_two_sided / 2 if t_statistic > 0 else 1 - (p_value_two_sided / 2)
                else:
                    # Single fold: no statistical test possible
                    t_statistic = np.nan
                    p_value = np.nan
                
                # Effect size (one-sample Cohen's d)
                effect_size = cohens_d_one_sample(run_values, baseline_value)
                
                # Determine improvement
                improvement = run_mean - baseline_value
                
                statistical_results.append({
                    'run': run,
                    'run_display': run_display,
                    'threshold': threshold_type,
                    'metric': metric_base,
                    'baseline_value': baseline_value,
                    'run_mean': run_mean,
                    'run_std': run_std,
                    'ci_lower': ci_low,
                    'ci_upper': ci_high,
                    'ci_method': ci_method,
                    'n_folds': n_folds,
                    't_statistic': t_statistic,
                    'p_value': p_value,
                    'cohens_d': effect_size,
                    'improvement': improvement,
                    'improvement_pct': (improvement / baseline_value * 100) if baseline_value > 0 else np.nan,
                })
    
    df_stats = pd.DataFrame(statistical_results)
    
    # Apply FDR correction for multiple comparisons
    if len(df_stats) > 0:
        if per_threshold_fdr:
            # Compute FDR separately for each threshold family
            for threshold in ['optimal', 'balanced']:
                subset_mask = (df_stats['threshold'] == threshold) & df_stats['p_value'].notna()
                if subset_mask.sum() > 0:
                    q_vals, sig = fdr_correction(
                        df_stats.loc[subset_mask, 'p_value'].values, 
                        alpha
                    )
                    df_stats.loc[subset_mask, 'q_value_fdr'] = q_vals
                    df_stats.loc[subset_mask, 'significant_fdr'] = sig
        else:
            # Global FDR across all tests
            valid_p = df_stats['p_value'].notna()
            if valid_p.sum() > 0:
                q_vals, sig = fdr_correction(
                    df_stats.loc[valid_p, 'p_value'].values,
                    alpha
                )
                df_stats.loc[valid_p, 'q_value_fdr'] = q_vals
                df_stats.loc[valid_p, 'significant_fdr'] = sig
        
        # Initialize columns with defaults if not present
        if 'q_value_fdr' not in df_stats.columns:
            df_stats['q_value_fdr'] = np.nan
        if 'significant_fdr' not in df_stats.columns:
            df_stats['significant_fdr'] = False
        
        # Also add uncorrected significance for reference
        df_stats['significant_uncorrected'] = df_stats['p_value'] < alpha
        
        # Significance markers based on FDR-corrected q-values
        def get_sig_marker(row):
            if pd.isna(row.get('q_value_fdr')):
                return ''
            q = row['q_value_fdr']
            if q < 0.001:
                return '***'
            elif q < 0.01:
                return '**'
            elif q < 0.05:
                return '*'
            else:
                return 'ns'
        
        df_stats['sig_marker'] = df_stats.apply(get_sig_marker, axis=1)
    
    return df_stats

# -------------------- LOAD METRICS + CLASS COUNTS ----------------------

def collect_metrics(results_base_dir: str) -> pd.DataFrame:
    base = Path(results_base_dir)
    files = list(base.rglob("metrics_summary.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics_summary.csv under {results_base_dir}")
    rows = []
    for f in files:
        rf = parse_run_fold_from_path(f)
        if not rf:
            print(f"[warn] cannot parse run/fold for {f}")
            continue
        run, fold = rf
        df = safe_read_csv(f)
        if df is None:
            print(f"[warn] empty/bad metrics file: {f}")
            continue
        rec = {"run": run, "fold": int(fold), "__metrics_path__": str(f)}
        rec.update(df.iloc[0].to_dict())
        rows.append(rec)
    results = pd.DataFrame(rows)
    results = to_numeric(results, METRICS_OF_INTEREST)
    results["run_display"] = results["run"].map(rename_run)
    return results

def infer_source_runs_root(results_base_dir: str) -> Optional[Path]:
    p = Path(results_base_dir)
    if "taguchi_runs_GRADCAM_RESULTS" in str(p):
        return Path(str(p).replace("taguchi_runs_GRADCAM_RESULTS", "taguchi_runs"))
    cand = p.parent / "taguchi_runs"
    return cand if cand.exists() else None

def read_val_counts_from_run_summary(source_runs_root: Path, run: str, fold: int) -> Optional[Tuple[int,int]]:
    """
    Read validation set counts from either run_summary.txt or fold_summary.txt.
    Tries both files and returns the first one that contains valid data.
    
    Args:
        source_runs_root: Root directory containing run folders
        run: Run identifier (e.g., "run_01")
        fold: Fold number
    
    Returns:
        Tuple of (risky_count, safe_count) or None if not found
    """
    # Try both possible file locations
    file_candidates = [
        source_runs_root / run / f"fold_{fold}" / "run_summary.txt",
        source_runs_root / run / f"fold_{fold}" / "fold_summary.txt"
    ]
    
    for sp in file_candidates:
        if not sp.exists():
            continue
        
        try:
            txt = sp.read_text()
            m1 = _VAL_RISKY_PAT.search(txt)
            m0 = _VAL_SAFE_PAT.search(txt)
            if m1 and m0:
                return int(m1.group(1)), int(m0.group(1))
        except Exception as e:
            print(f"[warn] Error reading {sp}: {e}")
            continue
    
    return None

# ------------------------ CM RECONSTRUCTION ----------------------------

def reconstruct_cm_from_metrics(P: int, N: int, precision_risky: float, recall_risky: float) -> Optional[np.ndarray]:
    """
    Using P (#positive=risky in val), N (#negative=safe in val),
    risky precision/recall → TP,FP,FN,TN (rounded, with small corrections).
    
    Returns:
        2x2 numpy array [[TP, FN], [FP, TN]] or None if invalid inputs
    """
    if not np.isfinite(precision_risky) or not np.isfinite(recall_risky):
        return None
    if P <= 0 or N < 0:
        return None
    
    # compute (float)
    tp = recall_risky * P
    fp = tp * (1.0 / max(precision_risky, 1e-12) - 1.0)
    fn = P - tp
    tn = N - fp
    
    # round to nearest int
    tp_i = int(round(tp)); fp_i = int(round(fp)); fn_i = int(round(fn)); tn_i = int(round(tn))
    
    # clamp small negatives due to rounding
    tp_i = max(0, tp_i); fp_i = max(0, fp_i); fn_i = max(0, fn_i); tn_i = max(0, tn_i)
    
    # enforce totals exactly
    dP = (tp_i + fn_i) - P
    if dP != 0:
        fn_i = max(0, fn_i - dP)
        dP2 = (tp_i + fn_i) - P
        if dP2 != 0:
            tp_i = max(0, tp_i - dP2)
    
    dN = (fp_i + tn_i) - N
    if dN != 0:
        tn_i = max(0, tn_i - dN)
        dN2 = (fp_i + tn_i) - N
        if dN2 != 0:
            fp_i = max(0, fp_i - dN2)
    
    # final safety
    tp_i = min(P, tp_i); fn_i = P - tp_i
    fp_i = min(N, fp_i); tn_i = N - fp_i
    
    return np.array([[tp_i, fn_i],
                     [fp_i, tn_i]], dtype=float)

def validate_cm_reconstruction(cm: np.ndarray, 
                               expected_precision: float, 
                               expected_recall: float,
                               tolerance: float = CM_RECONSTRUCTION_TOLERANCE) -> Dict[str, Any]:
    """
    Validate that reconstructed CM produces metrics within tolerance of expected values.
    
    Returns:
        Dict with validation results including 'valid' boolean and error metrics
    """
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]
    
    computed_metrics = metrics_from_counts(tp, fp, tn, fn)
    
    prec_error = abs(computed_metrics['precision'] - expected_precision) if np.isfinite(expected_precision) else 0
    rec_error = abs(computed_metrics['recall'] - expected_recall) if np.isfinite(expected_recall) else 0
    
    # Check if within tolerance
    prec_valid = prec_error <= tolerance or not np.isfinite(expected_precision)
    rec_valid = rec_error <= tolerance or not np.isfinite(expected_recall)
    
    return {
        'valid': prec_valid and rec_valid,
        'precision_error': prec_error,
        'recall_error': rec_error,
        'reconstructed_precision': computed_metrics['precision'],
        'reconstructed_recall': computed_metrics['recall'],
        'expected_precision': expected_precision,
        'expected_recall': expected_recall
    }

def metrics_from_counts(tp, fp, tn, fn):
    tp, fp, tn, fn = map(float, (tp, fp, tn, fn))
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1   = (2*prec*rec)/(prec+rec) if np.isfinite(prec) and np.isfinite(rec) and (prec+rec)>0 else np.nan
    acc  = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else np.nan
    return dict(precision=prec, recall=rec, f1=f1, accuracy=acc)

# ------------------------ PLOTTING HELPERS -----------------------------

def plot_cm(cm: np.ndarray, title: str, outpath: Path, clean_title: bool = True,
            normalize: str = "row", annot: str = "percent"):
    """
    Plot confusion matrix with optional normalization and annotation styles.
    
    Args:
        cm: Confusion matrix array
        title: Plot title
        outpath: Output file path
        clean_title: Whether to clean title for LaTeX
        normalize: 'none' | 'row' | 'col' | 'all' (default: 'row' for recall-based view)
        annot: 'percent' | 'count' | 'both' (default: 'percent')
    """
    if clean_title:
        title = title.replace("_", " ").strip()

    M = cm.astype(float)
    
    # Check if matrix is already normalized (all values in [0,1])
    is_already_normalized = (M.max() <= 1.0 and M.min() >= 0.0)
    
    # --- Normalization for display ---
    if normalize == "row":
        denom = M.sum(axis=1, keepdims=True)
        M_norm = np.divide(M, denom, out=np.zeros_like(M), where=denom > 0)
    elif normalize == "col":
        denom = M.sum(axis=0, keepdims=True)
        M_norm = np.divide(M, denom, out=np.zeros_like(M), where=denom > 0)
    elif normalize == "all":
        s = M.sum()
        M_norm = M / s if s > 0 else np.zeros_like(M)
    else:  # "none"
        M_norm = M

    # Choose what the heatmap colors represent and colorbar settings
    if normalize != "none":
        heat_values = M_norm
        cbar_label = 'Proportion'
        vmin, vmax = 0.0, 1.0
    elif is_already_normalized:
        # normalize="none" but matrix is already normalized (e.g., macro-avg)
        heat_values = M
        cbar_label = 'Proportion'
        vmin, vmax = 0.0, 1.0
    else:
        # Raw counts
        heat_values = M
        cbar_label = 'Count'
        vmin, vmax = None, None

    plt.figure(figsize=(5.8, 4.6))
    ax = sns.heatmap(
        heat_values, annot=False, cmap="Blues",
        xticklabels=["Pred Risky", "Pred Safe"],
        yticklabels=["Actual Risky", "Actual Safe"],
        cbar_kws={'label': cbar_label},
        vmin=vmin, vmax=vmax
    )

    # --- Custom annotations ---
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            parts = []
            if annot in ("percent", "both"):
                if normalize != "none":
                    # Already normalized by our code above
                    parts.append(f"{M_norm[i, j]:.1%}")
                elif is_already_normalized:
                    # Matrix came in already normalized (macro-avg)
                    parts.append(f"{M[i, j]:.1%}")
                else:
                    # Raw counts, divide by sum for percentage
                    parts.append(f"{(M[i, j] / M.sum()):.1%}")
            if annot in ("count", "both"):
                parts.append(f"({int(M[i, j])})")
            ax.text(j + 0.5, i + 0.5, "\n".join(parts),
                    ha="center", va="center", color="black", fontsize=11)

    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Rotate x-labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # Make y-labels horizontal to prevent overlap
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.savefig(outpath.with_suffix('.pdf'), bbox_inches="tight")
    plt.close()

def _highlight_best_bar(ax, values, mode="max"):
    """Highlight best bar with gold border"""
    try:
        vals = np.array(values, dtype=float)
        idx = np.nanargmin(vals) if mode=="min" else np.nanargmax(vals)
        ax.patches[idx].set_edgecolor("gold")
        ax.patches[idx].set_linewidth(3)
    except Exception:
        pass

def _order_with_baseline_first(series: pd.Series) -> pd.Categorical:
    """Order categorical with baseline first, then alphabetically"""
    unique_vals = series.unique()
    other_runs = sorted([x for x in unique_vals if x != BASELINE_NAME])
    categories = [BASELINE_NAME] + other_runs if BASELINE_NAME in unique_vals else other_runs
    return pd.Categorical(series, categories=categories, ordered=True)

# --------------------------- MAIN LOGIC --------------------------------

def compute_run_stats(results: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    """Compute statistics per run across folds with configurable CI level"""
    run_stats_list = []  # Renamed from 'stats' to avoid shadowing scipy_stats
    
    for run in sorted(results["run"].unique(), 
                     key=lambda r: (0,int(_RUN_PAT.search(r).group(1))) if _RUN_PAT.search(r) else (1,r)):
        rd = results[results["run"]==run]
        row = {"run": run, "run_display": rename_run(run), "num_folds": len(rd)}
        
        for m in METRICS_OF_INTEREST:
            if m in rd.columns:
                col = pd.to_numeric(rd[m], errors="coerce")
                row[f"{m}_mean"] = col.mean()
                row[f"{m}_std"] = col.std()
                
                # Add CI with configurable level
                if len(col.dropna()) > 1:
                    ci_low, ci_high = scipy_stats.t.interval(
                        ci_level,
                        len(col.dropna())-1,
                        loc=col.mean(), 
                        scale=scipy_stats.sem(col.dropna())
                    )
                    row[f"{m}_ci_lower"] = ci_low
                    row[f"{m}_ci_upper"] = ci_high
        
        run_stats_list.append(row)
    
    return pd.DataFrame(run_stats_list)

def add_baseline_rows(stats_df: pd.DataFrame) -> pd.DataFrame:
    base_m = metrics_from_counts(**_baseline_counts_lower())
    
    # Calculate safe recall (TNR) from baseline
    tn = BASELINE_CM["TN"]
    fp = BASELINE_CM["FP"]
    safe_recall = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    row = {
        "run": "__baseline__", "run_display": BASELINE_NAME, "num_folds": 1,
        "opt_risky_precision_mean": base_m["precision"],
        "opt_risky_recall_mean": base_m["recall"],
        "opt_risky_f1_mean": base_m["f1"],
        "opt_accuracy_mean": base_m["accuracy"],
        "opt_safe_recall_mean": safe_recall,
        "std_risky_precision_mean": base_m["precision"],
        "std_risky_recall_mean": base_m["recall"],
        "std_risky_f1_mean": base_m["f1"],
        "std_accuracy_mean": base_m["accuracy"],
        "std_safe_recall_mean": safe_recall,
    }
    out = pd.concat([pd.DataFrame([row]), stats_df], ignore_index=True)
    return out

def find_curve_files(metrics_path: Path, globs: List[str]) -> List[Path]:
    root = metrics_path.parent
    found = []
    for pat in globs:
        found.extend(root.rglob(pat))
    return list(sorted(set(found)))

def load_pr_points(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    df = safe_read_csv(path)
    if df is None: return None
    cols = [c.lower() for c in df.columns]
    if "precision" in cols and "recall" in cols:
        p = df.iloc[:, cols.index("precision")].values.astype(float)
        r = df.iloc[:, cols.index("recall")].values.astype(float)
        return r, p
    if "prec" in cols and "rec" in cols:
        p = df.iloc[:, cols.index("prec")].values.astype(float)
        r = df.iloc[:, cols.index("rec")].values.astype(float)
        return r, p
    return None

def load_roc_points(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    df = safe_read_csv(path)
    if df is None: return None
    cols = [c.lower() for c in df.columns]
    if "fpr" in cols and "tpr" in cols:
        fpr = df.iloc[:, cols.index("fpr")].values.astype(float)
        tpr = df.iloc[:, cols.index("tpr")].values.astype(float)
        return fpr, tpr
    return None

def average_curves(curves: List[Tuple[np.ndarray,np.ndarray]], xgrid: np.ndarray) -> Optional[Tuple[np.ndarray,np.ndarray]]:
    if not curves: return None
    ys = []
    for x, y in curves:
        idx = np.argsort(x)
        x_sorted, y_sorted = x[idx], y[idx]
        try:
            y_int = np.interp(xgrid, x_sorted, y_sorted, left=np.nan, right=np.nan)
        except Exception:
            continue
        ys.append(y_int)
    if not ys: return None
    ys = np.vstack(ys)
    with np.errstate(invalid="ignore"):
        y_mean = np.nanmean(ys, axis=0)
    return xgrid, y_mean

def plot_pr_roc_averaged(per_points: Dict[str, Dict[str, List[Tuple[np.ndarray,np.ndarray]]]],
                         outdir: Path):
    """Generate averaged PR and ROC curves and save curve points"""
    outdir.mkdir(parents=True, exist_ok=True)
    
    # ROC overlay
    xgrid = np.linspace(0,1,500)
    plt.figure(figsize=(6,5))
    for label in ["risky","safe"]:
        roc_curves = per_points.get("roc",{}).get(label, [])
        avg = average_curves(roc_curves, xgrid)
        if avg:
            x, y = avg
            plt.plot(x, y, label=label.title(), linewidth=2)
            # Save averaged curve points
            np.savetxt(
                outdir / f"roc_avg_{label}.csv",
                np.c_[x, y],
                delimiter=",",
                header="fpr,tpr",
                comments=""
            )
    plt.plot([0,1],[0,1], ls="--", color='gray', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "ROC.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "ROC.pdf", bbox_inches="tight")
    plt.close()

    # PR overlay
    xgrid = np.linspace(0,1,500)
    plt.figure(figsize=(6,5))
    for label in ["risky","safe"]:
        pr_curves = per_points.get("pr",{}).get(label, [])
        avg = average_curves(pr_curves, xgrid)
        if avg:
            r, p = avg
            plt.plot(r, p, label=label.title(), linewidth=2)
            # Save averaged curve points
            np.savetxt(
                outdir / f"pr_avg_{label}.csv",
                np.c_[r, p],
                delimiter=",",
                header="recall,precision",
                comments=""
            )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "PR.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "PR.pdf", bbox_inches="tight")
    plt.close()

# ------------------------- CONFUSION PIPELINE --------------------------

def reconstruct_all_confusions(results: pd.DataFrame,
                               source_runs_root: Path) -> Tuple[Dict[str, Dict[str, Dict[int, np.ndarray]]], List[Dict]]:
    """
    Return confusions[family][run][fold] -> 2x2 cm [[TP,FN],[FP,TN]]
    family ∈ {"optimal","balanced"} built from metrics + run_summary.txt or fold_summary.txt.
    
    Also returns validation_results list with reconstruction quality checks.
    """
    confusions = {"optimal": {}, "balanced": {}}
    validation_results = []
    
    for _, r in results.iterrows():
        run = r["run"]
        fold = int(r["fold"])
        
        counts = read_val_counts_from_run_summary(source_runs_root, run, fold)
        if counts is None:
            print(f"[warn] missing run_summary.txt or fold_summary.txt for {run}/fold_{fold}")
            continue
        P, N = counts

        # optimal family
        pr = float(r.get("opt_risky_precision", np.nan))
        rr = float(r.get("opt_risky_recall", np.nan))
        cm_opt = reconstruct_cm_from_metrics(P, N, pr, rr)
        if cm_opt is not None:
            confusions["optimal"].setdefault(run, {})[fold] = cm_opt
            # Validate reconstruction
            validation = validate_cm_reconstruction(cm_opt, pr, rr)
            validation.update({'run': run, 'fold': fold, 'family': 'optimal'})
            validation_results.append(validation)
            if not validation['valid']:
                print(f"[WARN] CM reconstruction error for {run}/fold_{fold} (optimal): "
                      f"prec_err={validation['precision_error']:.4f}, "
                      f"rec_err={validation['recall_error']:.4f}")

        # balanced family
        prb = float(r.get("std_risky_precision", np.nan))
        rrb = float(r.get("std_risky_recall", np.nan))
        cm_bal = reconstruct_cm_from_metrics(P, N, prb, rrb)
        if cm_bal is not None:
            confusions["balanced"].setdefault(run, {})[fold] = cm_bal
            # Validate reconstruction
            validation = validate_cm_reconstruction(cm_bal, prb, rrb)
            validation.update({'run': run, 'fold': fold, 'family': 'balanced'})
            validation_results.append(validation)
            if not validation['valid']:
                print(f"[WARN] CM reconstruction error for {run}/fold_{fold} (balanced): "
                      f"prec_err={validation['precision_error']:.4f}, "
                      f"rec_err={validation['recall_error']:.4f}")

    return confusions, validation_results

def sum_fold_cm(cm_dict: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    if not cm_dict: return None
    acc = None
    for cm in cm_dict.values():
        acc = cm if acc is None else acc + cm
    return acc

def macro_row_normalized_cm(foldmap: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    """
    Compute macro-average of row-normalized CMs across folds.
    
    This reflects "typical fold" behavior without letting any single fold dominate
    due to class imbalance. Each fold's CM is row-normalized first, then averaged.
    
    Returns:
        Array of shape (2, 2) with averaged row-normalized values, or None if empty
    """
    if not foldmap:
        return None
    
    mats = []
    for cm in foldmap.values():
        M = cm.astype(float)
        row_sums = M.sum(axis=1, keepdims=True)
        M_norm = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)
        mats.append(M_norm)
    
    return np.mean(np.stack(mats, axis=0), axis=0) if mats else None

def draw_and_save_all_cms(confusions, base_out: Path, run_names_map: Dict[str,str]):
    """Save all confusion matrices in organized structure with normalized percentages"""
    for fam in ["optimal","balanced"]:
        per_fold_dir = base_out / f"ConfusionMatrices/{fam.capitalize()}/PerFold"
        summed_dir = base_out / f"ConfusionMatrices/{fam.capitalize()}/Summed"
        
        for run, foldmap in confusions[fam].items():
            rlabel = run_names_map.get(run, run)
            
            # Save per-fold CMs with row normalization (recall-based view)
            for fold, cm in foldmap.items():
                out = per_fold_dir / f"{rlabel}_fold{fold}.png"
                plot_cm(cm, f"{rlabel} - Fold {fold}", out, normalize="row", annot="percent")
            
            # Save summed CM (micro: pooled behavior over all folds)
            summed = sum_fold_cm(foldmap)
            if summed is not None:
                out = summed_dir / f"{rlabel}_summed.png"
                plot_cm(summed, f"{rlabel} - Aggregated", out, normalize="row", annot="percent")
            
            # Save macro-average CM (average of per-fold row-normalized CMs)
            macro = macro_row_normalized_cm(foldmap)
            if macro is not None:
                out = summed_dir / f"{rlabel}_macroAvg.png"
                # macro is already row-normalized; plot with normalize="none" so colors match percents directly
                plot_cm(macro, f"{rlabel} - Macro Average", out, normalize="none", annot="percent")
    
    # Save baseline CM
    baseline_cm_array = np.array([
        [BASELINE_CM['TP'], BASELINE_CM['FN']],
        [BASELINE_CM['FP'], BASELINE_CM['TN']]
    ])
    baseline_out = base_out / "ConfusionMatrices/Baseline/C3D_Baseline.png"
    plot_cm(baseline_cm_array, "C3D Baseline", baseline_out, normalize="row", annot="percent")

def micro_metrics_from_cm(cm: np.ndarray) -> Dict[str,float]:
    tp, fn = cm[0,0], cm[0,1]
    fp, tn = cm[1,0], cm[1,1]
    return metrics_from_counts(tp, fp, tn, fn)

def prepare_cm_component_table(confusions, run_names_map, include_baseline=True) -> pd.DataFrame:
    """
    Prepare table with CM components and derived metrics (micro = from summed CM).
    """
    rows = []
    for fam in ["optimal","balanced"]:
        for run, foldmap in confusions[fam].items():
            summed = sum_fold_cm(foldmap)
            if summed is None: continue
            tp, fn = float(summed[0,0]), float(summed[0,1])
            fp, tn = float(summed[1,0]), float(summed[1,1])
            m = micro_metrics_from_cm(summed)
            rows.append(dict(
                threshold=fam, 
                run=run, 
                run_display=run_names_map.get(run, run),
                TP=tp, FP=fp, TN=tn, FN=fn,
                micro_precision=m["precision"], 
                micro_recall=m["recall"],
                micro_f1=m["f1"], 
                micro_accuracy=m["accuracy"]
            ))
    
    df = pd.DataFrame(rows)
    
    if include_baseline:
        for fam in ["balanced","optimal"]:
            base_row = dict(
                threshold=fam, 
                run="__baseline__", 
                run_display=BASELINE_NAME,
                TP=BASELINE_CM["TP"], 
                FP=BASELINE_CM["FP"], 
                TN=BASELINE_CM["TN"], 
                FN=BASELINE_CM["FN"]
            )
            bm = metrics_from_counts(**_baseline_counts_lower())
            
            # Add safe recall (TNR)
            tn = BASELINE_CM["TN"]
            fp = BASELINE_CM["FP"]
            safe_recall = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            
            base_row.update({
                'micro_precision': bm['precision'],
                'micro_recall': bm['recall'],
                'micro_f1': bm['f1'],
                'micro_accuracy': bm['accuracy'],
                'micro_safe_recall': safe_recall
            })
            df = pd.concat([pd.DataFrame([base_row]), df], ignore_index=True)
    
    return df

# ------------------------------- PLOTS ---------------------------------

def plot_cm_components_bar(df_comp: pd.DataFrame, outdir: Path, fam: str):
    d = df_comp[df_comp["threshold"]==fam].copy().sort_values("run_display")
    measures = [("TP","max"), ("FP","min"), ("TN","max"), ("FN","min")]
    plt.figure(figsize=(12,8))
    for i,(col,mode) in enumerate(measures,1):
        ax = plt.subplot(2,2,i)
        ax.bar(d["run_display"], d[col])
        _highlight_best_bar(ax, d[col].values, mode=mode)
        ax.set_title(f"{col} (Higher is {'Better' if mode=='max' else 'Worse'})")
        ax.set_xticklabels(d["run_display"], rotation=45, ha="right")
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "CM_components.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "CM_components.pdf", bbox_inches="tight")
    plt.close()

def plot_metric_bars(stats_with_baseline: pd.DataFrame, outdir: Path, fam: str):
    prefix = "opt_" if fam=="optimal" else "std_"
    labels = ["accuracy","risky_precision","risky_recall","risky_f1"]
    d = stats_with_baseline.copy()
    cols = [f"{prefix}{x}_mean" for x in labels]
    keep = ["run_display"] + cols
    d = d[keep].rename(columns={cols[i]:labels[i].replace("_"," ").title() for i in range(4)})
    d = d.sort_values("run_display")
    
    plt.figure(figsize=(12,8))
    for i,metric in enumerate(["Accuracy","Risky Precision","Risky Recall","Risky F1"],1):
        ax = plt.subplot(2,2,i)
        vals = d[metric].values
        ax.bar(d["run_display"], vals)
        _highlight_best_bar(ax, vals, mode="max")
        ax.set_title(metric)
        ax.set_xticklabels(d["run_display"], rotation=45, ha="right")
        ax.set_ylim(0,1.05)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "Metric_bars.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "Metric_bars.pdf", bbox_inches="tight")
    plt.close()

def plot_cm_boxplots(all_fold_cms, outdir: Path, fam: str):
    rows=[]
    for run, fmap in all_fold_cms[fam].items():
        for fold, cm in fmap.items():
            tp, fn = cm[0,0], cm[0,1]
            fp, tn = cm[1,0], cm[1,1]
            rows.append(dict(run=rename_run(run), fold=fold, TP=tp, FP=fp, TN=tn, FN=fn))
    if not rows: return
    df = pd.DataFrame(rows)
    
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df[["TP","FP","TN","FN"]])
    plt.title("Confusion Matrix Components Distribution Across Folds")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "CM_boxplots.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "CM_boxplots.pdf", bbox_inches="tight")
    plt.close()

def plot_heatmap(stats_with_baseline: pd.DataFrame, outdir: Path, fam: str):
    prefix = "opt_" if fam=="optimal" else "std_"
    rows = ["accuracy","risky_precision","risky_recall","risky_f1","safe_recall"]
    M = []
    for r in rows:
        col = f"{prefix}{r}_mean"
        M.append(stats_with_baseline[col].values)
    M = np.array(M, dtype=float)
    runs = stats_with_baseline["run_display"].values
    
    plt.figure(figsize=(max(10, 0.7*len(runs)), 6))
    ax = sns.heatmap(M, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                     yticklabels=[x.replace("_"," ").title() for x in rows],
                     xticklabels=runs, cbar_kws={"label":"Score"})
    
    # Make y-labels horizontal to prevent overlap
    plt.setp(ax.get_yticklabels(), rotation=0)
    # Rotate x-labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Outline best per metric row
    for i in range(M.shape[0]):
        try:
            j = int(np.nanargmax(M[i]))
            ax.add_patch(plt.Rectangle((j,i), 1,1, fill=False, edgecolor='black', linewidth=2))
        except Exception:
            pass
    
    plt.title("Performance Heatmap")
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "Performance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "Performance_heatmap.pdf", bbox_inches="tight")
    plt.close()

def copy_best_cms(confusions, stats_with_baseline, base_out: Path):
    """Copy best confusion matrices based on different metrics with normalized display"""
    for fam, prefix in [("optimal","opt_"), ("balanced","std_")]:
        for metric in ["accuracy","risky_precision","risky_recall","risky_f1"]:
            col = f"{prefix}{metric}_mean"
            d = stats_with_baseline[stats_with_baseline["run"]!="__baseline__"]
            if col not in d.columns or d[col].isna().all():
                continue
            best_idx = d[col].idxmax()
            best_run = d.loc[best_idx, "run"]
            best_display = d.loc[best_idx, "run_display"]
            summed = sum_fold_cm(confusions[fam].get(best_run, {}))
            if summed is None: continue
            
            out = base_out / f"ConfusionMatrices/Best/{fam.capitalize()}_{metric.title()}.png"
            plot_cm(summed, f"{best_display}", out, normalize="row", annot="percent")

def copy_performance_pngs_if_no_curves(results: pd.DataFrame, outdir: Path, fam: str):
    """Copy existing performance PNGs when curve CSVs are missing"""
    copied_count = 0
    for _, row in results.iterrows():
        mp = Path(row["__metrics_path__"])
        images_dir = mp.parent / "images"
        if images_dir.exists():
            for png in images_dir.glob("performance*.png"):
                # Include threshold family in filename for clarity
                run_fold_name = f"{fam}_{row['run']}_fold{row['fold']}"
                dst = outdir / f"{run_fold_name}_{png.name}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(png, dst)
                    copied_count += 1
                except Exception as e:
                    print(f"[warn] Could not copy {png}: {e}")
    return copied_count

# ------------------------------- BEST RUN TESTING --------------------------

def test_best_runs_vs_baseline(results: pd.DataFrame, baseline_metrics: Dict[str, float],
                                alpha: float = 0.05) -> pd.DataFrame:
    """
    Identify best run for each key metric and test if significantly better than baseline.
    
    Tests the following metrics:
    - Risky Recall
    - Risky Precision  
    - Risky F1
    - Accuracy
    - Safe Recall
    
    Returns DataFrame with best run info and statistical test results.
    """
    best_run_tests = []
    
    # Define metrics to test for both threshold types
    metrics_to_test = [
        ('risky_recall', 'recall'),
        ('risky_precision', 'precision'),
        ('risky_f1', 'f1'),
        ('accuracy', 'accuracy'),
        ('safe_recall', 'safe_recall')
    ]
    
    for threshold_type in ['optimal', 'balanced']:
        prefix = 'opt_' if threshold_type == 'optimal' else 'std_'
        
        for metric_col_suffix, baseline_key in metrics_to_test:
            metric_col = f'{prefix}{metric_col_suffix}'
            
            if metric_col not in results.columns:
                continue
            
            # Find best run for this metric
            run_means = results.groupby('run')[metric_col].mean()
            if run_means.empty or run_means.isna().all():
                continue
            
            best_run = run_means.idxmax()
            best_mean = run_means.max()
            
            # Get all fold values for best run
            best_run_data = results[results['run'] == best_run]
            run_values = best_run_data[metric_col].dropna().values
            
            if len(run_values) == 0:
                continue
            
            # Get baseline value
            baseline_value = baseline_metrics.get(baseline_key, np.nan)
            if not np.isfinite(baseline_value):
                continue
            
            # Calculate statistics
            run_std = np.std(run_values, ddof=1)
            n_folds = len(run_values)
            
            # Confidence interval
            if n_folds > 1:
                ci_low, ci_high = scipy_stats.t.interval(
                    1 - alpha,
                    n_folds - 1,
                    loc=best_mean,
                    scale=scipy_stats.sem(run_values)
                )
            else:
                ci_low = ci_high = best_mean
            
            # Single-sided t-test: H0: mean <= baseline, H1: mean > baseline
            if n_folds > 1:
                try:
                    res = scipy_stats.ttest_1samp(run_values, baseline_value, alternative='greater')
                    t_statistic = res.statistic
                    p_value = res.pvalue
                except TypeError:
                    t_statistic, p_value_two = scipy_stats.ttest_1samp(run_values, baseline_value)
                    p_value = p_value_two / 2 if t_statistic > 0 else 1 - (p_value_two / 2)
            else:
                t_statistic = np.nan
                p_value = np.nan
            
            # Effect size
            effect_size = cohens_d_one_sample(run_values, baseline_value)
            
            # Improvement
            improvement = best_mean - baseline_value
            
            # Determine significance
            is_significant = p_value < alpha if np.isfinite(p_value) else False
            
            best_run_tests.append({
                'threshold': threshold_type,
                'metric': metric_col_suffix.replace('_', ' ').title(),
                'best_run': best_run,
                'best_run_display': rename_run(best_run),
                'baseline_value': baseline_value,
                'best_mean': best_mean,
                'best_std': run_std,
                'ci_lower': ci_low,
                'ci_upper': ci_high,
                'n_folds': n_folds,
                't_statistic': t_statistic,
                'p_value': p_value,
                'cohens_d': effect_size,
                'improvement': improvement,
                'improvement_pct': (improvement / baseline_value * 100) if baseline_value > 0 else np.nan,
                'significant': is_significant,
                'sig_marker': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            })
    
    return pd.DataFrame(best_run_tests)

# ------------------------------- MAIN ----------------------------------

def main(results_base_dir: str, output_dir: Optional[str], 
         source_runs_dir: Optional[str], alpha: float = 0.05,
         use_bootstrap: bool = False, n_bootstrap: int = 10000,
         seed: Optional[int] = None,
         baseline_tp: Optional[int] = None, baseline_fp: Optional[int] = None,
         baseline_tn: Optional[int] = None, baseline_fn: Optional[int] = None,
         per_threshold_fdr: bool = True):
    
    # Allow CLI override of baseline CM
    global BASELINE_CM
    if all(x is not None for x in [baseline_tp, baseline_fp, baseline_tn, baseline_fn]):
        BASELINE_CM = dict(TP=baseline_tp, FP=baseline_fp, TN=baseline_tn, FN=baseline_fn)
        print(f"Using CLI-provided baseline CM: {BASELINE_CM}")
    
    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        print(f"Random seed set to: {seed}")
    
    results_base = Path(results_base_dir)
    out = Path(output_dir or (results_base / "CONSOLIDATED_RESULTS"))
    ensure_dirs(out)

    # Source runs
    if source_runs_dir:
        source_root = Path(source_runs_dir)
    else:
        source_root = infer_source_runs_root(results_base_dir)
        if not source_root or not source_root.exists():
            print("[warn] Could not infer source_runs_dir; using default path")
            source_root = Path("/ocean/projects/asc180003p/szaidi/Tackle_Ablation/taguchi_runs")

    print("=" * 70)
    print("CONSOLIDATING TAGUCHI RESULTS WITH STATISTICAL ANALYSIS")
    print("=" * 70)
    print(f"FDR method: {'per-threshold' if per_threshold_fdr else 'global'}")
    print(f"CI method: {'bootstrap' if use_bootstrap else 't-distribution'}")
    if not HAS_STATSMODELS:
        print("Note: Using fallback FDR implementation (statsmodels not available)")
    
    print("\n[1/9] Collecting metrics from all runs...")
    results = collect_metrics(results_base_dir)
    print(f"      Found {len(results)} fold results across {results['run'].nunique()} runs")

    print("\n[2/9] Reconstructing confusion matrices with validation...")
    confusions, validation_results = reconstruct_all_confusions(results, source_root)
    
    # Save validation results
    if validation_results:
        val_df = pd.DataFrame(validation_results)
        val_df.to_csv(out/"Tables"/"cm_reconstruction_validation.csv", index=False)
        n_invalid = (~val_df['valid']).sum()
        if n_invalid > 0:
            print(f"      WARNING: {n_invalid}/{len(val_df)} CM reconstructions exceeded tolerance")
            print(f"      See Tables/cm_reconstruction_validation.csv for details")

    print("\n[3/9] Computing statistics and adding baseline...")
    run_stats = compute_run_stats(results, ci_level=1 - alpha)
    baseline_metrics = metrics_from_counts(**_baseline_counts_lower())
    
    # Add safe recall (TNR) to baseline metrics
    tn = BASELINE_CM["TN"]
    fp = BASELINE_CM["FP"]
    baseline_metrics['safe_recall'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    stats_bl = add_baseline_rows(run_stats)
    stats_bl.to_csv(out/"Tables"/"summary_statistics.csv", index=False)
    print(f"      Saved: Tables/summary_statistics.csv")

    print(f"\n[4/9] Performing statistical significance testing (alpha={alpha})...")
    if use_bootstrap:
        print(f"      Using bootstrap CI with {n_bootstrap} resamples")
    
    stat_results = statistical_comparison(
        results, baseline_metrics, alpha=alpha,
        use_bootstrap=use_bootstrap, n_bootstrap=n_bootstrap, seed=seed,
        per_threshold_fdr=per_threshold_fdr
    )
    stat_results.to_csv(out/"Tables"/"Statistical"/"statistical_tests.csv", index=False)
    print(f"      Saved: Tables/Statistical/statistical_tests.csv")
    
    # Test best runs for key metrics
    print(f"\n      Testing best runs for key metrics vs C3D baseline...")
    best_run_tests = test_best_runs_vs_baseline(results, baseline_metrics, alpha=alpha)
    best_run_tests.to_csv(out/"Tables"/"Statistical"/"best_run_tests.csv", index=False)
    print(f"      Saved: Tables/Statistical/best_run_tests.csv")
    
    # Display best run results
    print(f"\n      BEST RUN ANALYSIS (Single-Sided t-test vs C3D Baseline):")
    print(f"      " + "=" * 65)
    for threshold in ['optimal', 'balanced']:
        print(f"\n      [{threshold.upper()}] Best Performers:")
        subset = best_run_tests[best_run_tests['threshold'] == threshold]
        for _, row in subset.iterrows():
            sig_note = f" ({row['sig_marker']})" if row['sig_marker'] != 'ns' else " (not sig.)"
            print(f"        {row['metric']:20s}: {row['best_run_display']:30s}")
            print(f"          → mean={row['best_mean']:.3f}, baseline={row['baseline_value']:.3f}, "
                  f"Δ={row['improvement']:+.3f} ({row['improvement_pct']:+.1f}%)")
            print(f"          → p={row['p_value']:.4f}{sig_note}, d={row['cohens_d']:.2f}")
    
    # Find best run for each threshold type
    for threshold in ['optimal', 'balanced']:
        print(f"\n      [{threshold.upper()}] Best performers vs C3D Baseline (all runs FDR-corrected):")
        subset = stat_results[stat_results['threshold'] == threshold]
        # FIX: Use 'f1' not 'risky_f1' - metric column stores base names
        for metric in ['accuracy', 'f1']:
            metric_subset = subset[subset['metric'] == metric].sort_values('run_mean', ascending=False)
            if not metric_subset.empty:
                best = metric_subset.iloc[0]
                print(f"        {metric:20s}: {best['run_display']:30s} "
                      f"(mean={best['run_mean']:.3f}, baseline={best['baseline_value']:.3f}, "
                      f"Δ={best['improvement']:+.3f}, "
                      f"p={best['p_value']:.4f}, q={best.get('q_value_fdr', np.nan):.4f} {best['sig_marker']}, "
                      f"d={best['cohens_d']:.2f})")

    print("\n[5/9] Saving confusion matrices...")
    run_names_map = {r: rename_run(r) for r in results["run"].unique()}
    draw_and_save_all_cms(confusions, out, run_names_map)
    print(f"      Saved: ConfusionMatrices/")

    # CM components table with micro metrics
    cm_comp = prepare_cm_component_table(confusions, run_names_map, include_baseline=True)
    cm_comp.to_csv(out/"Tables"/"cm_components_micro.csv", index=False)
    print(f"      Saved: Tables/cm_components_micro.csv")

    print("\n[6/9] Generating PR/ROC curves...")
    for fam in ["optimal","balanced"]:
        per_points = {"pr":{"risky":[], "safe":[]}, "roc":{"risky":[], "safe":[]}}
        for _, row in results.iterrows():
            mp = Path(row["__metrics_path__"])
            for p in find_curve_files(mp, [g for g in PR_GLOBS if "risky" in g]):
                pts = load_pr_points(p)
                if pts: per_points["pr"]["risky"].append(pts)
            for p in find_curve_files(mp, [g for g in PR_GLOBS if "safe" in g]):
                pts = load_pr_points(p)
                if pts: per_points["pr"]["safe"].append(pts)
            for p in find_curve_files(mp, [g for g in ROC_GLOBS if "risky" in g]):
                pts = load_roc_points(p)
                if pts: per_points["roc"]["risky"].append(pts)
            for p in find_curve_files(mp, [g for g in ROC_GLOBS if "safe" in g]):
                pts = load_roc_points(p)
                if pts: per_points["roc"]["safe"].append(pts)

        fam_curves_dir = out / f"Curves/{fam.capitalize()}"
        if any(per_points[k][c] for k in per_points for c in per_points[k]):
            plot_pr_roc_averaged(per_points, fam_curves_dir)
            print(f"      Saved: Curves/{fam.capitalize()}/ (from CSV points)")
        else:
            # Copy performance PNGs as fallback
            n_copied = copy_performance_pngs_if_no_curves(results, fam_curves_dir, fam)
            if n_copied > 0:
                print(f"      Copied {n_copied} performance PNGs to Curves/{fam.capitalize()}/")

    print("\n[7/9] Creating performance plots...")
    for fam in ["optimal","balanced"]:
        fam_plot_dir = out / f"Plots/{fam.capitalize()}"
        plot_cm_components_bar(cm_comp, fam_plot_dir, fam=fam)
        plot_metric_bars(stats_bl, fam_plot_dir, fam=fam)
        plot_cm_boxplots(confusions, fam_plot_dir, fam=fam)
        plot_heatmap(stats_bl, fam_plot_dir, fam=fam)
        print(f"      Saved: Plots/{fam.capitalize()}/")

    print("\n[8/9] Copying best confusion matrices...")
    copy_best_cms(confusions, stats_bl, out)
    print(f"      Saved: ConfusionMatrices/Best/")

    # Create summary report
    print("\n[9/9] Generating statistical summary report...")
    
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("STATISTICAL SUMMARY (FDR-CORRECTED)")
    summary_lines.append("=" * 70)
    summary_lines.append(f"FDR method: {'per-threshold families' if per_threshold_fdr else 'global across all tests'}")
    summary_lines.append(f"Significance level: α = {alpha}")
    summary_lines.append(f"CI level: {(1 - alpha):.0%}")
    summary_lines.append(f"CI method: {'bootstrap (n=' + str(n_bootstrap) + ')' if use_bootstrap else 't-distribution'}")
    
    # Add best run tests section
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append("BEST RUN ANALYSIS (vs C3D Baseline)")
    summary_lines.append("=" * 70)
    summary_lines.append("\nSingle-sided t-test: H0: run ≤ baseline, H1: run > baseline")
    summary_lines.append("(No FDR correction - testing pre-selected best runs per metric)")
    
    for threshold in ['optimal', 'balanced']:
        summary_lines.append(f"\n{threshold.upper()} THRESHOLD - Best Runs:")
        summary_lines.append("-" * 70)
        subset = best_run_tests[best_run_tests['threshold'] == threshold]
        
        for _, row in subset.iterrows():
            summary_lines.append(f"\n  {row['metric']}:")
            summary_lines.append(f"    Best Run: {row['best_run_display']}")
            summary_lines.append(
                f"    Performance: {row['best_mean']:.3f} ± {row['best_std']:.3f} "
                f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            )
            summary_lines.append(
                f"    Baseline: {row['baseline_value']:.3f}"
            )
            summary_lines.append(
                f"    Improvement: {row['improvement']:+.3f} ({row['improvement_pct']:+.1f}%)"
            )
            summary_lines.append(
                f"    Statistics: t={row['t_statistic']:.2f}, p={row['p_value']:.4f} {row['sig_marker']}, "
                f"d={row['cohens_d']:.2f}"
            )
    
    # Add full statistical results section
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append("ALL RUNS COMPARISON (FDR-Corrected)")
    summary_lines.append("=" * 70)
    
    for threshold in ['optimal', 'balanced']:
        summary_lines.append(f"\n{threshold.upper()} THRESHOLD:")
        summary_lines.append("-" * 70)
        subset = stat_results[(stat_results['threshold'] == threshold) & 
                             (stat_results['metric'].isin(['accuracy', 'precision', 'recall', 'f1']))]
        
        for run in subset['run_display'].unique():
            if run == BASELINE_NAME:
                continue
            run_data = subset[subset['run_display'] == run]
            summary_lines.append(f"\n  {run}:")
            for _, row in run_data.iterrows():
                q_val = row.get('q_value_fdr', np.nan)
                q_str = f"q={q_val:.4f}" if np.isfinite(q_val) else "q=N/A"
                summary_lines.append(
                    f"    {row['metric']:20s}: {row['run_mean']:.3f} ± {row['run_std']:.3f} "
                    f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
                    f"(baseline: {row['baseline_value']:.3f}, p={row['p_value']:.4f}, "
                    f"{q_str} {row['sig_marker']}, d={row['cohens_d']:.2f})"
                )
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save summary to file
    with open(out / "STATISTICAL_SUMMARY.txt", "w") as f:
        f.write(summary_text)
    
    print("\n" + "=" * 70)
    print(f"All results saved to: {out}")
    print("=" * 70)
    print(f"\nKey outputs:")
    print(f"  - Best run tests: Tables/Statistical/best_run_tests.csv")
    print(f"  - Statistical tests (FDR-corrected): Tables/Statistical/statistical_tests.csv")
    print(f"  - CM reconstruction validation: Tables/cm_reconstruction_validation.csv")
    print(f"  - Summary statistics: Tables/summary_statistics.csv")
    print(f"  - Micro metrics: Tables/cm_components_micro.csv")
    print(f"  - Text summary: STATISTICAL_SUMMARY.txt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Consolidate Taguchi results with rigorous statistical analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python consolidate_taguchi_plus.py --results_dir /path/to/results
  
  # With custom baseline CM
  python consolidate_taguchi_plus.py --results_dir /path/to/results \\
      --baseline_tp 10 --baseline_fp 5 --baseline_tn 25 --baseline_fn 8
  
  # With bootstrap CI and reproducible seed
  python consolidate_taguchi_plus.py --results_dir /path/to/results \\
      --use_bootstrap --n_bootstrap 10000 --seed 42
  
  # Custom significance level with global FDR
  python consolidate_taguchi_plus.py --results_dir /path/to/results \\
      --alpha 0.01 --global_fdr

Dependencies (requirements.txt):
  numpy>=1.20.0
  pandas>=1.3.0
  scipy>=1.7.0
  matplotlib>=3.4.0
  seaborn>=0.11.0
  statsmodels>=0.13.0  # optional, fallback provided
        """
    )
    
    ap.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
                    help=f"Base directory with GRADCAM results (default: {DEFAULT_RESULTS_DIR})")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Output directory (default: <results_dir>/CONSOLIDATED_RESULTS)")
    ap.add_argument("--source_runs_dir", type=str, default=None,
                    help="Root where run_summary.txt/fold_summary.txt lives (default: auto-infer)")
    
    # Statistical parameters
    stat_group = ap.add_argument_group('Statistical Options')
    stat_group.add_argument("--alpha", type=float, default=0.05,
                           help="Significance level for statistical tests (default: 0.05)")
    stat_group.add_argument("--use_bootstrap", action="store_true",
                           help="Use bootstrap confidence intervals instead of t-based CI")
    stat_group.add_argument("--n_bootstrap", type=int, default=10000,
                           help="Number of bootstrap resamples (default: 10000)")
    stat_group.add_argument("--seed", type=int, default=RANDOM_SEED,
                           help=f"Random seed for reproducibility (default: {RANDOM_SEED})")
    stat_group.add_argument("--global_fdr", action="store_true",
                           help="Use global FDR correction instead of per-threshold families (default: per-threshold)")
    
    # Baseline CM overrides
    baseline_group = ap.add_argument_group('Baseline Confusion Matrix Override')
    baseline_group.add_argument("--baseline_tp", type=int, default=None,
                               help=f"True Positives (default: {BASELINE_CM['TP']})")
    baseline_group.add_argument("--baseline_fp", type=int, default=None,
                               help=f"False Positives (default: {BASELINE_CM['FP']})")
    baseline_group.add_argument("--baseline_tn", type=int, default=None,
                               help=f"True Negatives (default: {BASELINE_CM['TN']})")
    baseline_group.add_argument("--baseline_fn", type=int, default=None,
                               help=f"False Negatives (default: {BASELINE_CM['FN']})")
    
    args = ap.parse_args()
    
    main(
        results_base_dir=args.results_dir,
        output_dir=args.output_dir,
        source_runs_dir=args.source_runs_dir,
        alpha=args.alpha,
        use_bootstrap=args.use_bootstrap,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        baseline_tp=args.baseline_tp,
        baseline_fp=args.baseline_fp,
        baseline_tn=args.baseline_tn,
        baseline_fn=args.baseline_fn,
        per_threshold_fdr=not args.global_fdr  # Default True, unless --global_fdr flag
    )