import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

# 启用 usetex 以使用系统 LaTeX 渲染
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times']

# 字体设置
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
try:
    font_prop = FontProperties(fname=font_path, size=24)
except:
    font_prop = FontProperties(family='serif', size=24)

# Data
preprocessing_components = ['Video Segmentation', 'Extractor Selection', 'Feature Extraction', 'Filtering']
preprocessing_times = [0.5, 1.0, 12.0, 1.5]

query_components = ['Object Detection', 'Tracking', 'Feature Matching']
query_times = [12.5, 1.0, 1.5]

sns.set_context("paper")
sns.set_palette("Set1")
sns.set_color_codes()

# --- 样式参数 ---
fontsize = 36
bar_height = 0.65

# 调整画布尺寸：宽12，高8 (足够容纳上下两张图，同时保持横向的宽阔感)
figsize_vertical_stack = (12, 8)

# ==========================================
# Plot: Combined Breakdown (Vertical Stack)
# ==========================================
plt.figure(figsize=figsize_vertical_stack)

# --- Subplot 1 (Top): Preprocessing ---
plt.subplot(2, 1, 1) # 2行1列，第1个
colors_pre = ['#b6ccd8', '#8ab8d4', '#5a9fc7', '#00668c']
bars1 = plt.barh(np.arange(len(preprocessing_components)), preprocessing_times, 
                 height=bar_height, color=colors_pre, edgecolor='black', linewidth=1.2)

for i, (bar, time) in enumerate(zip(bars1, preprocessing_times)):
    width = bar.get_width()
    plt.text(width + 0.2, bar.get_y() + bar.get_height()/2.,
             f'{time}',
             ha='left', va='center', fontsize=fontsize-6, fontweight='bold')

plt.xlabel('Time (minutes)', fontproperties=font_prop, fontsize=fontsize)
plt.yticks(np.arange(len(preprocessing_components)), preprocessing_components, 
           fontproperties=font_prop, fontsize=fontsize-4)
plt.xticks(fontproperties=font_prop, fontsize=fontsize-4)
plt.xlim(0, 14)
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.gca().invert_yaxis()
plt.title('Preprocessing', fontproperties=font_prop, fontsize=fontsize)

# --- Subplot 2 (Bottom): Query Execution ---
plt.subplot(2, 1, 2) # 2行1列，第2个
colors_query = ['#b6ccd8', '#5a9fc7', '#00668c']
bars2 = plt.barh(np.arange(len(query_components)), query_times, 
                 height=bar_height, color=colors_query, edgecolor='black', linewidth=1.2)

for i, (bar, time) in enumerate(zip(bars2, query_times)):
    width = bar.get_width()
    plt.text(width + 0.2, bar.get_y() + bar.get_height()/2.,
             f'{time}',
             ha='left', va='center', fontsize=fontsize-6, fontweight='bold')

plt.xlabel('Time (minutes)', fontproperties=font_prop, fontsize=fontsize)
plt.yticks(np.arange(len(query_components)), query_components, 
           fontproperties=font_prop, fontsize=fontsize-4)
plt.xticks(fontproperties=font_prop, fontsize=fontsize-4)
plt.xlim(0, 14)
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.gca().invert_yaxis()
plt.title('Query Execution', fontproperties=font_prop, fontsize=fontsize)

# 调整布局
plt.tight_layout()
# 增加一点上下子图之间的间距，防止上面的X轴标签和下面的标题重叠
plt.subplots_adjust(hspace=0.6) 

plt.savefig('cost_breakdown_combined_vertical.png', bbox_inches='tight', dpi=300)

plt.close()
plt.cla()
plt.clf()
