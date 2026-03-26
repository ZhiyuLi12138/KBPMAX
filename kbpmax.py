#!/usr/bin/env python3
"""
G4BP Algorithm Implementation
==============================
Implements the G4BP greedy algorithm and experimental validation for
k-BP function maximization, as described in the paper "K-BP-FUNCTION".

Mathematical Background
-----------------------
- V: ground set with |V| = n, k: number of subsets
- (k+1)^V: set of k-tuples (X1,...,Xk) of pairwise disjoint subsets of V
- Vector representation: x in {0,1,...,k}^V where x(e)=i means e in X_i
- Marginal gain: Delta_{e,i} f(x) = f(x with e added to subset i) - f(x)
- k-submodular curvature: kappa_f = 1 - min gain ratio
- k-supermodular curvature: kappa_g = 1 - min gain ratio

Usage
-----
    python kbpmax.py                    # full run (validate + experiment)
    python kbpmax.py --mode validate    # small-instance validation only
    python kbpmax.py --mode experiment  # full parameter sweep
    python kbpmax.py --n 10 --C 5       # custom instance size
"""

import argparse
import itertools
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# KBPFunction
# ---------------------------------------------------------------------------

class KBPFunction:
    """
    k-BP function  h(x) = lambda * f(x) + (1 - lambda) * g(x)

    where
        f  is k-submodular   with theoretical curvature kappa_f = alpha
        g  is k-supermodular with theoretical curvature kappa_g = beta

    Parameters
    ----------
    n : int
        |V|, total number of ground elements.
    k : int
        Number of disjoint subsets.
    alpha : float
        Curvature parameter for f (kappa_f = alpha).  0 <= alpha <= 1.
    beta : float
        Curvature parameter for g (kappa_g = beta).  0 <= beta < 1.
    lambda_param : float
        Mixing weight lambda in [0, 1].
    C : int
        Capacity / scale parameter (default 10).

    Notes
    -----
    The ground set V is split into two equal halves:
        V1 = {v_1, ..., v_{n/2}}   (indices 0 .. n/2-1)
        V2 = {v_{n/2+1}, ..., v_n} (indices n/2 .. n-1)

    Weights (only for V1 elements):
        w_i = (1 - alpha/C)^i - (1 - alpha/C)^{i+1},  i = 1, ..., |V1|

    Function definitions (epsilon = 1e-5):
        f(x) = (C - alpha*|x cap V2|)/C * sum_{v_i in supp(x) cap V1} w_i
               + |x cap V2| / C

        g(x) = |x|/(1-beta) - beta*min(1+|x cap V1|, |x|, C)
               + eps * max(|x|, |x| + beta/(1-beta) * (|x cap V2| - C + 1))

        h(x) = lambda * f(x) + (1 - lambda) * g(x)
    """

    def __init__(self, n: int, k: int, alpha: float, beta: float,
                 lambda_param: float, C: int = 10):
        if n % 2 != 0:
            raise ValueError("n must be even so that V1 and V2 have equal size.")
        self.n = n
        self.k = k
        self.alpha = alpha
        # beta < 1 is required by the g function (1/(1-beta) term).
        # Values at or above 1 represent a degenerate limit; clamp to avoid
        # division by zero while preserving the mathematical behaviour.
        if beta >= 1.0:
            beta = 1.0 - 1e-9
        self.beta = beta
        self.lambda_param = lambda_param
        self.C = C
        self.epsilon = 1e-5

        half = n // 2
        self.V1 = list(range(half))
        self.V2 = list(range(half, n))
        self.V1_set = set(self.V1)
        self.V2_set = set(self.V2)

        # Weights w_i = (1 - alpha/C)^i - (1 - alpha/C)^{i+1}  for i=1,...,|V1|
        base = 1.0 - alpha / C if C > 0 else 1.0
        self.weights = np.array(
            [base ** i - base ** (i + 1) for i in range(1, half + 1)],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Core functions
    # ------------------------------------------------------------------

    def f(self, x: np.ndarray) -> float:
        """k-submodular function (theoretical curvature kappa_f = alpha)."""
        supp = set(int(e) for e in np.where(x != 0)[0])
        x_v2 = len(supp & self.V2_set)

        # Sum weights for V1 elements in support
        w_sum = sum(
            self.weights[j] for j, vi in enumerate(self.V1) if vi in supp
        )

        return (self.C - self.alpha * x_v2) / self.C * w_sum + x_v2 / self.C

    def g(self, x: np.ndarray) -> float:
        """k-supermodular function (theoretical curvature kappa_g = beta)."""
        supp = set(int(e) for e in np.where(x != 0)[0])
        x_size = len(supp)
        x_v1 = len(supp & self.V1_set)
        x_v2 = len(supp & self.V2_set)

        beta = self.beta
        term1 = x_size / (1.0 - beta)
        term2 = beta * min(1 + x_v1, x_size, self.C)
        inner = x_size + beta / (1.0 - beta) * (x_v2 - self.C + 1)
        term3 = self.epsilon * max(float(x_size), inner)

        return term1 - term2 + term3

    def h(self, x: np.ndarray) -> float:
        """BP function h = lambda*f + (1-lambda)*g."""
        lam = self.lambda_param
        return lam * self.f(x) + (1.0 - lam) * self.g(x)

    # ------------------------------------------------------------------
    # Marginal gains
    # ------------------------------------------------------------------

    def marginal_gain_f(self, x: np.ndarray, e: int, i: int) -> float:
        """Delta_{e,i} f(x).  Requires x[e] == 0."""
        x_new = x.copy()
        x_new[e] = i
        return self.f(x_new) - self.f(x)

    def marginal_gain_g(self, x: np.ndarray, e: int, i: int) -> float:
        """Delta_{e,i} g(x).  Requires x[e] == 0."""
        x_new = x.copy()
        x_new[e] = i
        return self.g(x_new) - self.g(x)

    def marginal_gain(self, x: np.ndarray, e: int, i: int) -> float:
        """Combined marginal gain for h.  Requires x[e] == 0."""
        x_new = x.copy()
        x_new[e] = i
        return self.h(x_new) - self.h(x)

    # ------------------------------------------------------------------
    # Curvature (theoretical)
    # ------------------------------------------------------------------

    def curvature_f(self) -> float:
        """Theoretical curvature kappa_f = alpha."""
        return self.alpha

    def curvature_g(self) -> float:
        """Theoretical curvature kappa_g = beta."""
        return self.beta


# ---------------------------------------------------------------------------
# G4BP Algorithm
# ---------------------------------------------------------------------------

class G4BP:
    """
    Algorithm 1: Greedy algorithm for k-BP function maximization.

    Parameters
    ----------
    func : KBPFunction
        The objective function.
    constraint_type : str
        'total'      – |supp(x)| <= C
        'individual' – |X_j| <= C_j for all j in [k]
    C : int
        Total capacity (used when constraint_type='total').
    C_list : list[int] or None
        Per-subset capacities [C_1,...,C_k] (used when
        constraint_type='individual').  If None, defaults to C//k each.
    """

    def __init__(self, func: KBPFunction, constraint_type: str = "total",
                 C: int = 10, C_list=None):
        self.func = func
        self.constraint_type = constraint_type
        self.C = C
        if C_list is None:
            self.C_list = [C // func.k] * func.k
        else:
            self.C_list = list(C_list)

    def _can_add(self, x: np.ndarray, e: int, i: int) -> bool:
        """
        Return True if adding element e to subset i keeps x feasible.

        Element e must not already be in the support of x.
        """
        if x[e] != 0:
            return False
        if self.constraint_type == "total":
            return int(np.sum(x != 0)) + 1 <= self.C
        else:
            return int(np.sum(x == i)) + 1 <= self.C_list[i - 1]

    def run(self):
        """
        Execute the G4BP greedy algorithm.

        Returns
        -------
        x_hat : np.ndarray, shape (n,), dtype int
            Solution vector.  x_hat[e] in {0,...,k} where 0 means not selected.
        h_value : float
            Objective value h(x_hat).
        """
        n = self.func.n
        k = self.func.k
        x = np.zeros(n, dtype=int)
        R = set(range(n))          # remaining (unassigned) elements

        while True:
            best_gain = -float("inf")
            best_e = None
            best_i = None

            for e in R:
                for j in range(1, k + 1):
                    if self._can_add(x, e, j):
                        gain = self.func.marginal_gain(x, e, j)
                        if gain > best_gain:
                            best_gain = gain
                            best_e = e
                            best_i = j

            if best_e is None:
                break   # no feasible move exists

            x[best_e] = best_i
            R.discard(best_e)

        return x, self.func.h(x)


# ---------------------------------------------------------------------------
# OPT computation
# ---------------------------------------------------------------------------

def exhaustive_search(func: KBPFunction, constraint_type: str = "total",
                      C: int = 10, C_list=None):
    """
    Find the exact optimum by exhaustive enumeration.

    WARNING: Only feasible for small instances.
    For k=3 the state space is (k+1)^n = 4^n states:
      n= 8 →      65,536 states  (fast, < 1 s)
      n=10 →   1,048,576 states  (seconds)
      n=12 →  16,777,216 states  (tens of seconds)
      n=20 →  ~1 × 10^12 states  (infeasible — use random_sampling_opt)

    Returns
    -------
    best_x : np.ndarray
    best_val : float
    """
    n = func.n
    k = func.k
    C_list_eff = C_list if C_list is not None else [C // k] * k
    best_val = -float("inf")
    best_x = np.zeros(n, dtype=int)

    for assignment in itertools.product(range(k + 1), repeat=n):
        x = np.array(assignment, dtype=int)

        if constraint_type == "total":
            if int(np.sum(x != 0)) > C:
                continue
        else:
            if any(int(np.sum(x == j)) > C_list_eff[j - 1] for j in range(1, k + 1)):
                continue

        val = func.h(x)
        if val > best_val:
            best_val = val
            best_x = x.copy()

    return best_x, best_val


def random_sampling_opt(func: KBPFunction, constraint_type: str = "total",
                        C: int = 10, C_list=None,
                        n_samples: int = 20000, seed: int = 42):
    """
    Estimate OPT by random sampling (lower bound on true OPT).

    Used when exhaustive search is infeasible (large n).

    Returns
    -------
    best_x : np.ndarray
    best_val : float
    """
    rng = np.random.default_rng(seed)
    n = func.n
    k = func.k
    C_list_eff = C_list if C_list is not None else [C // k] * k
    best_val = -float("inf")
    best_x = np.zeros(n, dtype=int)

    for _ in range(n_samples):
        x = np.zeros(n, dtype=int)
        perm = rng.permutation(n)

        if constraint_type == "total":
            n_sel = int(rng.integers(1, C + 1))
            selected = perm[:n_sel]
            for e in selected:
                x[e] = int(rng.integers(1, k + 1))
        else:
            for j in range(1, k + 1):
                cap = C_list_eff[j - 1]
                n_sel = int(rng.integers(0, cap + 1))
                avail = [e for e in perm if x[e] == 0]
                for e in avail[:n_sel]:
                    x[e] = j

        val = func.h(x)
        if val > best_val:
            best_val = val
            best_x = x.copy()

    return best_x, best_val


# ---------------------------------------------------------------------------
# Theoretical guarantee
# ---------------------------------------------------------------------------

def theoretical_guarantee(kappa_f: float, kappa_g: float) -> float:
    """
    Compute the theoretical approximation guarantee gamma:

        gamma = (1/kappa_f) * (1 - exp(-(1 - kappa_g) * kappa_f))  if kappa_f > 0
        gamma = 1                                                    if kappa_f = 0
    """
    if kappa_f <= 0.0:
        return 1.0
    return (1.0 / kappa_f) * (1.0 - np.exp(-(1.0 - kappa_g) * kappa_f))


# ---------------------------------------------------------------------------
# Validation on a small instance
# ---------------------------------------------------------------------------

def validate_small_instance(n_small: int = 8, C_small: int = 4, k: int = 3):
    """
    Uses exhaustive search ((k+1)^n states, e.g. 4^8 = 65,536 for k=3, n=8)
    to find true OPT and verifies that the G4BP approximation ratio is at
    least the theoretical guarantee.
    """
    print(f"\n{'='*64}")
    print(f"Validation: small instance  n={n_small}, k={k}, C={C_small}")
    print(f"{'='*64}")
    print(
        f"{'alpha':>6} {'beta':>6} {'lambda':>7} | "
        f"{'G4BP':>10} {'OPT':>10} {'ratio':>8} {'gamma':>8} {'ratio>=gamma':>12}"
    )
    print("-" * 70)

    test_cases = [
        (0.0, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (0.3, 0.0, 0.5),
        (0.3, 0.3, 0.5),
        (0.5, 0.5, 0.3),
        (0.5, 0.5, 0.7),
        (0.7, 0.3, 0.5),
        (0.3, 0.7, 0.5),
    ]

    all_pass = True
    for alpha, beta, lam in test_cases:
        func = KBPFunction(n=n_small, k=k, alpha=alpha, beta=beta,
                           lambda_param=lam, C=C_small)
        alg = G4BP(func, constraint_type="total", C=C_small)
        _, h_hat = alg.run()

        _, opt = exhaustive_search(func, constraint_type="total", C=C_small)

        ratio = h_hat / opt if opt > 1e-10 else 1.0
        gamma = theoretical_guarantee(alpha, beta)
        ok = ratio >= gamma - 1e-6    # small tolerance for float arithmetic

        if not ok:
            all_pass = False

        print(
            f"{alpha:>6.1f} {beta:>6.1f} {lam:>7.2f} | "
            f"{h_hat:>10.4f} {opt:>10.4f} {ratio:>8.4f} {gamma:>8.4f} "
            f"{'PASS' if ok else 'FAIL':>12}"
        )

    print("-" * 70)
    print(f"All cases pass: {all_pass}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def experiment(n: int = 20, k: int = 3, C: int = 10,
               use_exhaustive: bool = False, n_samples: int = 20000):
    """
    Parameter sweep over alpha, beta, lambda in {0, 0.1, ..., 1.0}.

    For each combination:
      - Run G4BP to obtain h_hat.
      - Estimate OPT via random sampling (or exhaustive if use_exhaustive=True).
      - Compute approximation ratio and theoretical guarantee.

    Returns
    -------
    results : dict
        Key: (alpha, beta, lambda_param) tuple.
        Value: dict with keys 'h_hat', 'opt', 'ratio', 'gamma'.
    """
    param_values = np.round(np.arange(0.0, 1.01, 0.1), 1)
    total = len(param_values) ** 3
    print(f"\n{'='*64}")
    print(f"Experiment: n={n}, k={k}, C={C}, {total} parameter combos")
    print(f"OPT method: {'exhaustive' if use_exhaustive else 'random sampling'}")
    print(f"{'='*64}")

    results = {}
    count = 0

    for alpha in param_values:
        for beta in param_values:
            for lam in param_values:
                func = KBPFunction(n=n, k=k, alpha=float(alpha),
                                   beta=float(beta), lambda_param=float(lam),
                                   C=C)

                # G4BP
                alg = G4BP(func, constraint_type="total", C=C)
                _, h_hat = alg.run()

                # OPT
                if use_exhaustive:
                    _, opt = exhaustive_search(func, constraint_type="total", C=C)
                else:
                    # Collision-resistant seed derived from all three parameters.
                    seed = (int(round(alpha * 1000)) * 1_000_000
                            + int(round(beta * 1000)) * 1_000
                            + int(round(lam * 1000)))
                    _, opt = random_sampling_opt(
                        func, constraint_type="total", C=C,
                        n_samples=n_samples,
                        seed=seed,
                    )

                ratio = h_hat / opt if opt > 1e-10 else 1.0
                gamma = theoretical_guarantee(float(alpha), float(beta))

                results[(float(alpha), float(beta), float(lam))] = {
                    "h_hat": h_hat,
                    "opt":   opt,
                    "ratio": ratio,
                    "gamma": gamma,
                }

                count += 1
                if count % 200 == 0:
                    print(f"  {count}/{total} completed …")

    print(f"  {count}/{total} completed.")
    print("Experiment complete!")
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(results: dict, save_dir: str = "results"):
    """
    Generate and save three figures:

    1. heatmap_ratio.png  – heatmaps of approximation ratio for selected
                            lambda values (x-axis alpha, y-axis beta).
    2. theory_vs_actual.png – scatter: theoretical guarantee vs actual ratio.
    3. ratio_vs_beta.png  – line plot of ratio and gamma vs beta for
                            several lambda values at fixed alpha=0.3.
    """
    os.makedirs(save_dir, exist_ok=True)
    param_values = np.round(np.arange(0.0, 1.01, 0.1), 1)

    # Ratios above 1.0 can occur when random sampling underestimates OPT;
    # cap display at this value so the scatter plot remains readable.
    DISPLAY_RATIO_CAP = 1.2

    # ------------------------------------------------------------------ #
    # Figure 1: Heatmaps
    # ------------------------------------------------------------------ #
    lambda_show = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(lambda_show), figsize=(22, 4))
    fig.suptitle(r"Approximation Ratio  $h(\hat{x})$ / OPT", fontsize=14)

    for ax, lam_target in zip(axes, lambda_show):
        # Snap to nearest param value
        lam_key = float(min(param_values, key=lambda v: abs(v - lam_target)))

        ratio_mat = np.zeros((len(param_values), len(param_values)))
        for i, beta in enumerate(param_values):
            for j, alpha in enumerate(param_values):
                key = (float(alpha), float(beta), lam_key)
                if key in results:
                    ratio_mat[i, j] = min(results[key]["ratio"], 1.0)

        sns.heatmap(
            ratio_mat, ax=ax,
            xticklabels=[f"{a:.1f}" for a in param_values],
            yticklabels=[f"{b:.1f}" for b in param_values],
            vmin=0, vmax=1, cmap="YlOrRd_r",
            cbar=True, linewidths=0.0,
        )
        ax.set_title(f"λ = {lam_key:.2f}", fontsize=11)
        ax.set_xlabel("α  (κ_f)", fontsize=9)
        ax.set_ylabel("β  (κ_g)", fontsize=9)
        ax.invert_yaxis()

    plt.tight_layout()
    path1 = os.path.join(save_dir, "heatmap_ratio.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path1}")

    # ------------------------------------------------------------------ #
    # Figure 2: Theoretical guarantee vs actual ratio
    # ------------------------------------------------------------------ #
    gammas = []
    ratios = []
    for val in results.values():
        gammas.append(val["gamma"])
        ratios.append(min(val["ratio"], DISPLAY_RATIO_CAP))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(gammas, ratios, alpha=0.25, s=8, c="steelblue",
               label="Experiments")
    lo, hi = 0.0, 1.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5,
            label=r"ratio = $\gamma$  (tight)")
    ax.set_xlabel(r"Theoretical guarantee $\gamma$", fontsize=12)
    ax.set_ylabel("Actual approximation ratio", fontsize=12)
    ax.set_title(r"G4BP: Actual vs Theoretical Approximation Ratio", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.02, 1.1)
    ax.set_ylim(-0.02, DISPLAY_RATIO_CAP + 0.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(save_dir, "theory_vs_actual.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path2}")

    # ------------------------------------------------------------------ #
    # Figure 3: Ratio vs beta for fixed alpha, several lambda values
    # ------------------------------------------------------------------ #
    alpha_fixed = 0.3
    alpha_key = float(min(param_values, key=lambda v: abs(v - alpha_fixed)))
    lambda_lines = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lambda_lines)))

    fig, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Approximation Ratio vs β  (α = {alpha_key:.1f})", fontsize=13)

    for lam_target, color in zip(lambda_lines, colors):
        lam_key = float(min(param_values, key=lambda v: abs(v - lam_target)))
        betas, ratios_line, gammas_line = [], [], []
        for beta in param_values:
            key = (alpha_key, float(beta), lam_key)
            if key in results:
                betas.append(float(beta))
                ratios_line.append(results[key]["ratio"])
                gammas_line.append(results[key]["gamma"])

        axes2[0].plot(betas, ratios_line, "-o", color=color, markersize=4,
                      label=f"λ={lam_key:.2f}")
        axes2[1].plot(betas, gammas_line, "-o", color=color, markersize=4,
                      label=f"λ={lam_key:.2f}")

    for ax_i, title in zip(axes2, ["Actual approximation ratio",
                                    "Theoretical guarantee γ"]):
        ax_i.set_xlabel("β  (κ_g curvature)", fontsize=11)
        ax_i.set_ylabel(title, fontsize=11)
        ax_i.set_title(title, fontsize=11)
        ax_i.legend(fontsize=8)
        ax_i.set_ylim(bottom=0)
        ax_i.grid(True, alpha=0.3)

    plt.tight_layout()
    path3 = os.path.join(save_dir, "ratio_vs_beta.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path3}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    ratios = [v["ratio"] for v in results.values()]
    gammas = [v["gamma"] for v in results.values()]
    n_below = sum(r < g - 1e-6 for r, g in zip(ratios, gammas))

    print("\n=== Summary ===")
    print(f"  Experiments         : {len(results)}")
    print(f"  Mean ratio          : {np.mean(ratios):.4f}")
    print(f"  Min  ratio          : {np.min(ratios):.4f}")
    print(f"  Max  ratio          : {np.max(ratios):.4f}")
    print(f"  Mean γ              : {np.mean(gammas):.4f}")
    print(f"  Cases ratio >= γ    : {len(ratios) - n_below}/{len(ratios)}")


def main():
    parser = argparse.ArgumentParser(
        description="G4BP algorithm experiments (K-BP-FUNCTION paper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "experiment", "full"],
        default="full",
        help=(
            "validate – small-instance exhaustive check only; "
            "experiment – full parameter sweep; "
            "full – both"
        ),
    )
    parser.add_argument("--n", type=int, default=20,
                        help="|V| for the main experiment (must be even).")
    parser.add_argument("--C", type=int, default=10,
                        help="Capacity constraint.")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of disjoint subsets.")
    parser.add_argument("--samples", type=int, default=20000,
                        help="Random samples per experiment for OPT estimation.")
    parser.add_argument("--exhaustive", action="store_true",
                        help="Use exact exhaustive search for OPT (only for small n).")
    parser.add_argument("--output", type=str, default="results",
                        help="Directory for output figures.")
    args = parser.parse_args()

    if args.mode in ("validate", "full"):
        validate_small_instance()

    if args.mode in ("experiment", "full"):
        results = experiment(
            n=args.n, k=args.k, C=args.C,
            use_exhaustive=args.exhaustive,
            n_samples=args.samples,
        )
        plot_results(results, save_dir=args.output)
        _print_summary(results)


if __name__ == "__main__":
    main()
