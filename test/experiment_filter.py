import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap

methods = ["Retrieval-based", "OMEGA w/o RL", "OMEGA w/o IL", "OMEGA"]
colors = ['#b6ccd8', '#8ab8d4', '#5a9fc7', '#00668c']
datasets = ["Seattle(Car)", "Seattle(Bus)", "Seattle(Truck)", "Youtubu-8M-pet", "Youtubu-8M-  mixture"]

F1 = np.array([
    [51.0, 62.5, 70.2, 78.0],
    [52.3, 63.8, 71.5, 78.5],
    [50.5, 61.8, 69.7, 77.8],
    [69.0, 71.8, 74.5, 77.0],
    [73.0, 75.2, 77.1, 79.0],
])

Time = np.array([
    [3526,  588,  320,  235],
    [2980,  497,  270,  198],
    [3210,  538,  292,  214],
    [2247,  593,  165,  119],
    [2134,  613,  175,  126],
])
F1 = F1 / 100.0
F1_err = None
Time_err = None

# =====================
# 1) 放大字号（统一风格）
# =====================
BASE = 16  # 你想更大就改成 15/16
plt.rcParams.update({
    "font.size": BASE,
    "axes.titlesize": BASE + 1,
    "axes.labelsize": BASE + 2,
    "xtick.labelsize": BASE - 1,
    "ytick.labelsize": BASE - 1,
    "legend.fontsize": BASE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =====================
# 2) 放大画布：原来(16, 5.2) -> 更大
# =====================
fig, axes = plt.subplots(2, 5, figsize=(20, 7.2))  # 更大：可试 (22, 8)

x = np.arange(len(methods))
bar_w = 0.75

for j, ds in enumerate(datasets):
    # --- 上排：F1 ---
    ax = axes[0, j]
    ax.bar(
        x, F1[j], width=bar_w, color=colors, edgecolor="none",
        yerr=(None if F1_err is None else F1_err[j]),
        capsize=3
    )

    # 标题太长就换行（每行最多约14个字符，你可调）
    title_ds = "\n".join(textwrap.wrap(ds, width=14))
    ax.set_title(f"{chr(ord('a') + j)}) {title_ds}", pad=6)

    if j == 0:
        ax.set_ylabel("Filtering F1")
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, 1.0)


    # --- 下排：Time ---
    ax2 = axes[1, j]
    ax2.bar(
        x, Time[j], width=bar_w, color=colors, edgecolor="none",
        yerr=(None if Time_err is None else Time_err[j]),
        capsize=3
    )
    if j == 0:
        ax2.set_ylabel("Filtering Time (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=18, ha="right")  # 字大了，少转一点也行
    ax2.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4)
    ax2.set_axisbelow(True)

# =====================
# 3) 全局图例：字也会跟 legend.fontsize 变大
# =====================
handles = [Rectangle((0, 0), 1, 1, color=c) for c in colors]
fig.legend(
    handles, methods,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.08)
)

# =====================
# 4) 留出顶部给图例（画布大了，间距也适当放松）
# =====================
fig.subplots_adjust(
    left=0.05, right=0.995,
    bottom=0.18, top=0.82,
    wspace=0.28, hspace=0.40
)

plt.savefig("omega_f1_time_2x5_big.png", dpi=300, bbox_inches="tight")
plt.show()
