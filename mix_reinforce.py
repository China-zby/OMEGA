import os
import cv2
import numpy as np
import random
import concurrent.futures
from tqdm import tqdm
from color_extractor import color_extraction
from energy_extractor import energy_extraction
from lbp_extractor import lbp_extraction
from net_extractor import net_extraction
from color_energy_extractor import color_energy_extraction
from color_lbp_extractor import color_lbp_extraction
from energy_lbp_extractor import glcm_lbp_extraction

# 定义参数
n_states = 50  # 状态空间大小
n_actions = 7  # 行动空间大小（4种特征提取方法）
epsilon = 0.5  # 探索概率
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子

# 初始化Q表
Q_TABLE_PATH = "q_table.npy"
# 初始化或加载Q表
if os.path.exists(Q_TABLE_PATH):
    Q = np.load(Q_TABLE_PATH)
    print("Loaded Q-Table from file.")
else:
    Q = np.zeros((n_states, n_actions))
    print("Initialized new Q-Table.")

# 选择特征提取方法
def select_feature_extraction_method(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, n_actions - 1)  # 探索
    else:
        return np.argmax(Q[state, :])  # 利用

# 奖励函数
def get_reward(recall, precision):
    if recall > 0.55:
        return precision
    else:
        return 0

# 更新Q表
def update_q_table(state, action, reward, next_state):
    Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

# 单个查询处理函数
def process_query(i,cache, query_path, video_path, query_num, total_frames, n_states, all_trajectories):
    # 初始化状态
    state = 0
    video_flag = [1] * (total_frames + 2)  # 根据具体需求设置
    total_reward = []
    total_action = "start"
    for episode in range(10):  # 训练轮数
        action = select_feature_extraction_method(state)

        if action == 0:
            recall, precision, video_flag, flag, count, cache = color_extraction(query_path, video_path, query_num, video_flag,cache)
        elif action == 1:
            recall, precision, video_flag, flag, count, cache = energy_extraction(query_path, video_path, query_num, video_flag, cache)
        elif action == 2:
            recall, precision, video_flag, flag, count, cache = lbp_extraction(query_path, video_path, query_num, video_flag, cache)
        elif action == 3:
            recall, precision, video_flag, flag, count, cache = color_lbp_extraction(query_path, video_path, query_num, video_flag, cache)
        elif action == 4:
            recall, precision, video_flag, flag, count, cache = color_energy_extraction(query_path, video_path, query_num, video_flag, cache)
        elif action == 5:
            recall, precision, video_flag, flag, count, cache = glcm_lbp_extraction(query_path, video_path, query_num, video_flag, cache)
        else:
            recall, precision, video_flag, flag, count, cache = net_extraction(query_path, video_path, query_num, video_flag, cache)

        total_action = total_action + "->" + str(action)
        if flag == 1:
            video_flag = [1] * (total_frames + 2)
            all_trajectories.append((str(i), total_action, total_reward))
            total_action = "start"
            total_reward = []
            continue
        reward = get_reward(recall[i], precision[i])
        if reward == 0:
            video_flag = [1] * (total_frames + 2)
            all_trajectories.append((str(i), total_action, total_reward))
            total_action = "start"
            total_reward = []
            continue
        total_reward.append((reward, count[i]))
        next_state = (state + 1) % n_states
        update_q_table(state, action, reward, next_state)
        state = next_state

    tqdm.write(f'Episode {episode + 1}, Query {i + 1}: Total Reward = {total_reward}')
    return all_trajectories

# 主函数
def main():
    query_path = '/home/ubuntu/zby/query_taipei_test'
    video_path = '/home/ubuntu/zby/taipei_640.mp4'
    query_num = 10
    cache={}
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_trajectories = []  # 用于记录所有探索轨迹

    # 使用多进程处理每个查询
    with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_query, i,cache, query_path, video_path, query_num, total_frames, n_states, all_trajectories) for i in range(query_num)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=query_num, desc="Processing Queries"):
            all_trajectories.extend(future.result())

    # 输出最终的Q表
    print("Final Q-Table:")
    print(Q)
    np.save(Q_TABLE_PATH, Q)
    print("Q-Table saved to file.")

    with open("trajectories_2.txt", "w") as f:
        for trajectory in all_trajectories:
            f.write(f"{trajectory}\n")
        f.write("\n")  # 分隔不同的探索轨迹
    print("Trajectories saved to file.")

if __name__ == "__main__":
    main()
