import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# X轴：video segment
segments = np.array([50, 100, 200, 500, 1000])

# OMEGA（随 segment 变化）
f1_omega = np.array([0.78, 0.79, 0.795, 0.792, 0.788])
t_omega  = np.array([140, 155, 175, 230, 310])  # s

# Retrieval-based（不变：常数 -> 复制成数组）
f1_retr_const = 0.75
t_retr_const  = 950  # s
f1_retr = np.full_like(segments, f1_retr_const, dtype=float)
t_retr  = np.full_like(segments, t_retr_const,  dtype=float)

# =====================
# 字体：特别大
# =====================
BASE = 22
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
# 画布：更紧凑，但两图拉开
# =====================
fig, axes = plt.subplots(
    1, 2,
    figsize=(9.6, 3.7),   # 比你原来(13.5, 4.8)紧凑很多
    sharex=True,
    constrained_layout=False
)

omega_color = "#00668c"
retr_color  = "#b55a30"

# OMEGA 圆点，Retrieval 方块（你现在的设定保留）
line_kw  = dict(linewidth=2.8, marker="o", markersize=8, markeredgewidth=1.6)
line_kw2 = dict(linewidth=2.8, marker="s", markersize=8, markeredgewidth=2.0)

# ---- 左：F1 ----
ax = axes[0]
ax.plot(segments, f1_omega, color=omega_color, label="OMEGA", **line_kw)
ax.plot(segments, f1_retr,  color=retr_color,  label="Retrieval-based", **line_kw2)

ax.set_title("Accuracy", pad=6)
ax.set_xlabel("Video segment (frames)", labelpad=6)
ax.set_ylabel("F1 Score", labelpad=6)

# 更好看的紧凑 y 范围（不想用就改回 0~1）
ymin = max(0.0, min(f1_omega.min(), f1_retr.min()) - 0.03)
ymax = min(1.0, max(f1_omega.max(), f1_retr.max()) + 0.03)
ax.set_ylim(ymin, ymax)

ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.35)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=BASE-8)


# ---- 右：Time ----
ax2 = axes[1]
ax2.plot(segments, t_omega, color=omega_color, label="OMEGA", **line_kw)
ax2.plot(segments, t_retr,  color=retr_color,  label="Retrieval-based", **line_kw2)

ax2.set_title("Efficiency", pad=6)
ax2.set_xlabel("Video segment (frames)", labelpad=6)
ax2.set_ylabel("Query Time (s)", labelpad=10)

ax2.grid(True, linestyle="--", linewidth=0.9, alpha=0.35)
ax2.set_axisbelow(True)

# 关键：右图 y 轴刻度减少，避免重叠
ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

# X轴：log + 显示原始数字（防重叠）
for a in axes:
    a.set_xscale("log")
    a.set_xticks(segments)
    a.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    a.ticklabel_format(axis="x", style="plain")
    a.tick_params(axis="both", which="major", length=6, width=1.4, pad=4)

# 两图离远些（关键）
fig.subplots_adjust(
    left=0.085,
    right=0.995,
    bottom=0.22,
    top=0.88,
    wspace=0.5   # 拉开；还想远就 0.42/0.45
)

plt.savefig("segment_sensitivity_baseline_bigfont_spaced.png", dpi=300, bbox_inches="tight")
plt.show()
