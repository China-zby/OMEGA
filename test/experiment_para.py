import numpy as np
import matplotlib.pyplot as plt

lambdas = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
f1 = np.array([0.729, 0.782, 0.776, 0.728, 0.692])
time_s = np.array([415, 297, 285, 268, 195])

# =====================
# 字体：特别大
# =====================
BASE = 22  # 你说“特别大”，直接从 22 起步；不够就 24/26
plt.rcParams.update({
    "font.size": BASE,
    "axes.titlesize": BASE + 4,
    "axes.labelsize": BASE + 4,
    "xtick.labelsize": BASE + 1,
    "ytick.labelsize": BASE + 1,
    "legend.fontsize": BASE + 1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =====================
# 画布：更紧凑
# =====================
fig, axes = plt.subplots(
    1, 2,
    figsize=(9.2, 3.6),   # 更紧凑（你原来是 12.5 x 4.8）
    sharex=True,
    constrained_layout=False  # 我们手动 subplots_adjust 更可控
)

# marker 改方形（你前面说要方形）
line_kw = dict(linewidth=2.8, marker="o", markersize=8, markeredgewidth=2.0)

# ---- 左图：F1 ----
ax = axes[0]
ax.plot(lambdas, f1, color="#00668c", **line_kw)

ax.set_title("Accuracy", pad=6)
ax.set_xlabel("Trade-off factor λ", labelpad=6)
ax.set_ylabel("F1 Score", labelpad=6)
ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.35)
ax.set_axisbelow(True)

# 更紧凑但更好看：不要 0~1 这么空，建议用数据上下留白
ymin = max(0.0, f1.min() - 0.03)
ymax = min(1.0, f1.max() + 0.03)
ax.set_ylim(ymin, ymax)

# ---- 右图：Time ----
ax2 = axes[1]
ax2.plot(lambdas, time_s, color="#5a9fc7", **line_kw)

ax2.set_title("Efficiency", pad=6)
ax2.set_xlabel("Trade-off factor λ", labelpad=6)
ax2.set_ylabel("Query Time (s)", labelpad=6)
ax2.grid(True, linestyle="--", linewidth=0.9, alpha=0.35)
ax2.set_axisbelow(True)

# X轴：log + 原始刻度显示
for a in axes:
    a.set_xscale("log")
    a.set_xticks(lambdas)
    a.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    a.ticklabel_format(axis="x", style="plain")
    a.tick_params(axis="both", which="major", length=6, width=1.4, pad=4)

# 标最佳 λ
best_lambda = 0.5
for a in axes:
    a.axvline(best_lambda, color="gray", linestyle=":", linewidth=2.0, alpha=0.85)

# =====================
# 手动压缩间距（关键）
# =====================
fig.subplots_adjust(
    left=0.075,   # 左边距小一点
    right=0.995,  # 右边距小一点
    bottom=0.22,  # 给大字号 x label 留空间
    top=0.88,     # 标题空间
    wspace=0.5   # 两张图之间更近
)

plt.savefig("lambda_sensitivity_compact_bigfont.png", dpi=300, bbox_inches="tight")
plt.show()
