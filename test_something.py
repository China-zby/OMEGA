import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_trajectory_figure():
    # 设置画布大小
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off') # 隐藏坐标轴

    # ==========================================
    # 1. 左侧：Query Image 区域
    # ==========================================
    # 画一个框代表 Query 图片位置
    query_box = patches.Rectangle((0.5, 2.5), 2.5, 2.5, linewidth=2, edgecolor='#333333', facecolor='#E0E0E0')
    ax.add_patch(query_box)
    ax.text(1.75, 3.75, "Paste\nQuery Image\nHere", ha='center', va='center', color='#666666', fontsize=12)
    
    # 图标：用户
    ax.text(1.75, 5.2, "User Query", ha='center', fontsize=14, fontweight='bold')

    # 箭头指向中间
    ax.arrow(3.2, 3.75, 1.0, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')

    # ==========================================
    # 2. 中间：Video Stream & Trajectory 区域
    # ==========================================
    # 画胶卷背景 (Film Strip)
    film_start_x = 4.5
    film_y = 2.0
    film_width = 7.0
    film_height = 3.5
    
    # 胶卷底色
    film_bg = patches.Rectangle((film_start_x, film_y), film_width, film_height, linewidth=0, facecolor='#F5F5F5', zorder=0)
    ax.add_patch(film_bg)
    
    # 画几帧 (Frames)
    num_frames = 4
    frame_w = 1.5
    gap = 0.2
    
    # 定义两个轨迹的坐标 (模拟马的移动)
    # 轨迹A (Target - Red): 从左下往右移动
    traj_a_coords = [(5.0, 2.5), (6.7, 2.6), (8.4, 2.4), (10.1, 2.5)] 
    # 轨迹B (Noise - Blue): 或者是另一匹马，位置不同
    traj_b_coords = [(5.2, 4.0), (6.5, 4.2), (8.6, 3.8), (10.3, 4.1)]

    for i in range(num_frames):
        x_pos = film_start_x + 0.2 + i * (frame_w + gap)
        # 画每一帧的边框
        frame = patches.Rectangle((x_pos, film_y + 0.5), frame_w, 2.5, linewidth=1, edgecolor='#999999', facecolor='white')
        ax.add_patch(frame)
        
        # 模拟胶卷孔
        ax.add_patch(patches.Rectangle((x_pos, film_y + 3.1), frame_w, 0.3, color='black'))
        ax.add_patch(patches.Rectangle((x_pos, film_y + 0.1), frame_w, 0.3, color='black'))
        
        # --- 画轨迹 A (Target) 的 Bounding Box ---
        bx, by = traj_a_coords[i]
        # 在这里你需要贴上马的小图，现在用红色框代替
        rect_a = patches.Rectangle((bx, by), 0.6, 0.8, linewidth=2, edgecolor='#D32F2F', facecolor='none', linestyle='-')
        ax.add_patch(rect_a)
        
        # --- 画轨迹 B (Noise) 的 Bounding Box ---
        bx2, by2 = traj_b_coords[i]
        # 用蓝色框代替干扰项
        rect_b = patches.Rectangle((bx2, by2), 0.5, 0.7, linewidth=2, edgecolor='#1976D2', facecolor='none', linestyle='--')
        ax.add_patch(rect_b)

    # --- 画连接线 (Trajectory Links) ---
    # 连线 A
    for i in range(len(traj_a_coords)-1):
        x1, y1 = traj_a_coords[i]
        x2, y2 = traj_a_coords[i+1]
        # 连接框的中心
        ax.plot([x1+0.3, x2+0.3], [y1+0.4, y2+0.4], color='#D32F2F', linewidth=2, alpha=0.5, linestyle='-')

    # 连线 B
    for i in range(len(traj_b_coords)-1):
        x1, y1 = traj_b_coords[i]
        x2, y2 = traj_b_coords[i+1]
        ax.plot([x1+0.25, x2+0.25], [y1+0.35, y2+0.35], color='#1976D2', linewidth=2, alpha=0.5, linestyle='--')

    ax.text(film_start_x + film_width/2, 1.5, "Video Stream & Trajectory Extraction", ha='center', fontsize=12, fontweight='bold')
    ax.text(film_start_x + film_width/2, 1.1, "(Spatial-Temporal Association)", ha='center', fontsize=10, color='#555555')

    # ==========================================
    # 3. 匹配逻辑图标
    # ==========================================
    # 漏斗/筛选图标
    ax.text(12.0, 3.75, "Match", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.arrow(11.6, 3.75, 0.8, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')

    # ==========================================
    # 4. 右侧：Output Target 区域
    # ==========================================
    output_x = 13.0
    
    # 结果框
    out_frame = patches.Rectangle((output_x, 2.5), 2.5, 2.5, linewidth=3, edgecolor='#D32F2F', facecolor='white')
    ax.add_patch(out_frame)
    
    # 里面画一个示意的轨迹
    ax.plot([output_x+0.5, output_x+1.2, output_x+2.0], [3.0, 3.2, 3.1], color='#D32F2F', linewidth=2, marker='o')
    
    ax.text(output_x + 1.25, 3.75, "Paste\nTarget Video\nClip Here", ha='center', va='center', color='#D32F2F', fontsize=10)
    ax.text(output_x + 1.25, 5.2, "Matched Trajectory", ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

draw_trajectory_figure()
