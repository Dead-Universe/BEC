import numpy as np
import pandas as pd

# Step 1: Generate 16-run LHS for CLRS hyperparameters (fixed random seed)
np.random.seed(2025)

n_runs = 16

# Parameter ranges (use variable names exactly as in your code, add paper variable notes)
params = {
    "rho_star": (0.75, 0.95),  # ρ* (target routing entropy)
    "k_fb": (0.05, 0.40),  # k_fb (feedback gain)
    "c_pulse": (0.20, 2.00),  # c_pulse (pulse gain)
    "tau_hi": (2.00, 3.00),  # τ_hi (high temperature baseline)
    "sigma_hi": (0.00, 0.06),  # σ_hi (high noise baseline)
    "phi1": (0.10, 0.25),  # φ₁ (explore fraction)
    "phi2": (0.55, 0.75),  # φ₂ (settle fraction)
    "lambda_aux": (0.005, 0.050),  # λ_aux (LBL weight)
}

df = pd.DataFrame({"run_id": np.arange(1, n_runs + 1)})

for name, (lo, hi) in params.items():
    bins = np.linspace(0, 1, n_runs + 1)
    u = bins[:-1] + (bins[1:] - bins[:-1]) * np.random.rand(n_runs)
    np.random.shuffle(u)
    values = lo + (hi - lo) * u
    df[name] = values

# Fixed parameters (remain constant across runs)
df["tau_lo"] = 1.00  # τ_lo
df["sigma_lo"] = 0.00  # σ_lo
df["alpha_ema"] = 0.10  # EMA smoothing
df["tau_min"] = 1.00  # temperature lower bound
df["tau_max"] = 3.00  # temperature upper bound
df["sigma_min"] = 0.00  # noise lower bound
df["sigma_max"] = 0.10  # noise upper bound

df["comment"] = "16-run LHS screening; vary core CLRS hyperparameters; others fixed"

# Save to CSV
path = "/mnt/data/clrs_sensitivity_plan_16runs.csv"
df.to_csv(path, index=False)
