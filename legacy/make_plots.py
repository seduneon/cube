"""Generate paper figures from evaluation results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

# ── Data from run ──────────────────────────────────────────────────────────────
MODELS     = ['EMLP-Col', 'EMLP', 'MLP-Aug']
COLORS_BAR = ['#2166ac', '#4dac26', '#f1a340']

MAE      = [0.932, 1.239, 2.356]
PEARSONR = [0.900, 0.839, 0.359]
ACC      = [45.4,  31.9,  12.1]

EQUIV_ERR = [8.2e-7, 5.0e-7, 0.776]   # moves

DEPTHS      = [4, 7, 10, 14]
SOLVE_RATES = {
    'EMLP-Col': [100.0, 100.0, 97.2, 14.5],
    'EMLP':     [100.0, 100.0, 94.2, 27.5],
    'MLP-Aug':  [ 99.6,  63.0,  1.4,  0.0],
}

# ── Helper ─────────────────────────────────────────────────────────────────────
def save(fig, name):
    path = f"{OUT}/{name}.pdf"
    fig.savefig(path, bbox_inches='tight')
    path2 = f"{OUT}/{name}.png"
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path2}")

# ── Fig 1: Solve rate by depth ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(DEPTHS))
n = len(MODELS)
width = 0.18
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

for i, (model, color) in enumerate(zip(MODELS, COLORS_BAR)):
    rates = SOLVE_RATES[model]
    bars = ax.bar(x + offsets[i], rates, width, label=model,
                  color=color, alpha=0.88, edgecolor='white', linewidth=0.5)

ax.set_xlabel('Scramble depth', fontsize=12)
ax.set_ylabel('Solve rate (%)', fontsize=12)
ax.set_title('Beam-search solve rate by scramble depth\n(beam width $W=20$)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(DEPTHS)
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(fontsize=10, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(100, color='gray', lw=0.8, ls='--', alpha=0.5)
fig.tight_layout()
save(fig, 'solve_rate_by_depth')

# ── Fig 2: Prediction metrics comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
x = np.arange(len(MODELS))
width = 0.55

for ax, values, ylabel, title, fmt in zip(
        axes,
        [MAE, PEARSONR, ACC],
        ['MAE (moves)', 'Pearson $r$', 'Rounded accuracy (%)'],
        ['Mean Absolute Error', 'Pearson Correlation', 'Rounded Accuracy'],
        ['{:.3f}', '{:.3f}', '{:.1f}%']):
    bars = ax.bar(x, values, width, color=COLORS_BAR, alpha=0.88,
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*ax.get_ylim()[1],
                fmt.format(val), ha='center', va='bottom', fontsize=8)

fig.suptitle('Value prediction metrics (test set)', fontsize=12, y=1.02)
fig.tight_layout()
save(fig, 'prediction_metrics')

# ── Fig 3: Equivariance error (log scale) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.8))
x = np.arange(len(MODELS))
bars = ax.bar(x, EQUIV_ERR, 0.55, color=COLORS_BAR, alpha=0.88,
              edgecolor='white', linewidth=0.5)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=10)
ax.set_ylabel('Equivariance error (moves, log scale)', fontsize=10)
ax.set_title('Spatial equivariance error\n(lower is better; equivariant models at float32 floor)', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# annotation
ax.axhline(1e-6, color='gray', lw=0.8, ls='--', alpha=0.6)
ax.text(len(MODELS)-0.5, 1.4e-6, 'float32 floor', color='gray', fontsize=8, ha='right')
fig.tight_layout()
save(fig, 'equivariance_error')

# ── Fig 4: Solve rate depth 10 summary bar ────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.5))
rates_d10 = [SOLVE_RATES[m][2] for m in MODELS]
bars = ax.barh(MODELS[::-1], rates_d10[::-1], color=COLORS_BAR[::-1],
               alpha=0.88, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Solve rate at depth 10 (%)', fontsize=11)
ax.set_title('Greedy solve rate at scramble depth 10\n(beam width $W = 20$)', fontsize=11)
ax.set_xlim(0, 110)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, rates_d10[::-1]):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=10)
fig.tight_layout()
save(fig, 'solve_rate_depth10')

print("All plots saved to", OUT)
