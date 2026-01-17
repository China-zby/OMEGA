# import glob
# import os
# import cv2
# import numpy as np
# import random
# import concurrent.futures
# import tensorflow as tf
# from tensorflow.keras import models, layers, optimizers
# from collections import deque
# from tqdm import tqdm
# from color_extractor import color_extraction
# from energy_extractor import energy_extraction
# from lbp_extractor import lbp_extraction
# from net_extractor import net_extraction
# from color_energy_extractor import color_energy_extraction
# from color_lbp_extractor import color_lbp_extraction
# from energy_lbp_extractor import glcm_lbp_extraction

# # 定义参数
# n_actions = 7  # 行动空间大小（7种特征提取方法）
# epsilon = 0.5  # 探索概率
# epsilon_min = 0.1
# epsilon_decay = 0.995
# gamma = 0.9  # 折扣因子
# learning_rate = 0.001
# batch_size = 32
# memory_size = 200
# n_states = 7

# # 初始化经验回放缓冲区
# memory = deque(maxlen=memory_size)

# # 定义策略网络
# def build_policy_model():
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(n_actions, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=learning_rate))
#     return model

# # 初始化策略网络
# policy_model = build_policy_model()
# target_model = build_policy_model()
# target_model.set_weights(policy_model.get_weights())

# # 选择特征提取方法
# def select_feature_extraction_method(state):
#     if np.random.rand() <= epsilon:
#         return random.randint(0, n_actions - 1)  # 探索
#     else:
#         prob = policy_model.predict(state)
#         return np.argmax(prob[0])  # 利用

# # 奖励函数
# def get_reward(recall, precision):
#     if recall > 0.55:
#         return precision
#     else:
#         return 0

# # 存储经验
# def store_memory(state, action, reward, next_state, done):
#     memory.append((state, action, reward, next_state, done))

# # 训练策略网络
# def train_model():
#     if len(memory) < batch_size:
#         return
#     batch = random.sample(memory, batch_size)
#     for state, action, reward, next_state, done in batch:
#         target = reward
#         if not done:
#             target = reward + gamma * np.amax(target_model.predict(next_state)[0])
#         target_f = policy_model.predict(state)
#         target_f[0][action] = target
#         policy_model.fit(state, target_f, epochs=1, verbose=0)

# # 更新目标网络
# def update_target_model():
#     target_model.set_weights(policy_model.get_weights())

# # 单个查询处理函数
# def process_query(i, query_path, video_path, query_num, total_frames, n_states, all_trajectories):
#     # 初始化状态
#     state = 0
#     video_flag = [1] * (total_frames + 2)  # 根据具体需求设置
#     total_reward = []
#     total_action = "start"
#     query_inputs = glob.glob(os.path.join(query_path, "*"))
#     query_image = cv2.imread(query_inputs[i])
#     if query_image is None:
#         print(f"Error: Could not read query image {i}.")
#         return all_trajectories
#     query_image_resized = cv2.resize(query_image, (64, 64))
#     query_state = np.expand_dims(query_image_resized, axis=0)  # 增加一个维度

#     for episode in range(3):  # 训练轮数
#         action = select_feature_extraction_method(query_state)

#         if action == 0:
#             recall, precision, video_flag, flag, count = color_extraction(query_path, video_path, query_num, video_flag)
#         elif action == 1:
#             recall, precision, video_flag, flag, count = energy_extraction(query_path, video_path, query_num, video_flag)
#         elif action == 2:
#             recall, precision, video_flag, flag, count = lbp_extraction(query_path, video_path, query_num, video_flag)
#         # elif action == 3:
#         #     recall, precision, video_flag, flag, count = color_lbp_extraction(query_path, video_path, query_num, video_flag)
#         # elif action == 4:
#         #     recall, precision, video_flag, flag, count = color_energy_extraction(query_path, video_path, query_num, video_flag)
#         # elif action == 5:
#         #     recall, precision, video_flag, flag, count = glcm_lbp_extraction(query_path, video_path, query_num, video_flag)
#         else:
#             recall, precision, video_flag, flag, count = net_extraction(query_path, video_path, query_num, video_flag)

#         total_action = total_action + "->" + str(action)
#         if flag == 1:
#             video_flag = [1] * (total_frames + 2)
#             all_trajectories.append((str(i), total_action, total_reward))
#             total_action = "start"
#             total_reward = []
#             continue
#         reward = get_reward(recall[i], precision[i])
#         if reward == 0:
#             video_flag = [1] * (total_frames + 2)
#             all_trajectories.append((str(i), total_action, total_reward))
#             total_action = "start"
#             total_reward = []
#             continue
#         total_reward.append((reward, count))
#         next_state = (state + 1) % n_states
#         store_memory(query_state, action, reward, next_state, flag == 1)
#         query_state = next_state

#         if len(memory) >= batch_size:
#             train_model()

#     tqdm.write(f'Episode {episode + 1}, Query {i + 1}: Total Reward = {total_reward}')
#     return all_trajectories

# # 主函数
# def main():
#     query_path = '/home/ubuntu/zby/query_taipei_test'
#     video_path = '/home/ubuntu/zby/taipei_640.mp4'
#     query_num = 20
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     all_trajectories = []  # 用于记录所有探索轨迹

#     # 使用多进程处理每个查询
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = [executor.submit(process_query, i, query_path, video_path, query_num, total_frames, n_states, all_trajectories) for i in range(query_num)]
#         for future in tqdm(concurrent.futures.as_completed(futures), total=query_num, desc="Processing Queries"):
#             all_trajectories.extend(future.result())

#     # 输出最终的策略模型
#     print("Final Policy Model:")
#     policy_model.save('policy_model.h5')
#     print("Policy model saved to file.")

#     with open("trajectories_1.txt", "w") as f:
#         for trajectory in all_trajectories:
#             f.write(f"{trajectory}\n")
#         f.write("\n")  # 分隔不同的探索轨迹
#     print("Trajectories saved to file.")

# if __name__ == "__main__":
#     main()

import glob
import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from collections import deque
import pickle


import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'memory'))
from tqdm import tqdm
from color_extractor import color_extraction
from energy_extractor import energy_extraction
from lbp_extractor import lbp_extraction
from net_extractor import net_extraction
from color_energy_extractor import color_energy_extraction
from color_lbp_extractor import color_lbp_extraction
from energy_lbp_extractor import glcm_lbp_extraction
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 定义参数
n_actions = 7  # 行动空间大小（7种特征提取方法）
epsilon = 0.5  # 探索概率
epsilon_min = 0.1
epsilon_decay = 0.995
gamma = 0.9  # 折扣因子
learning_rate = 0.001
batch_size = 4
memory_size = 20

# 初始化经验回放缓冲区
memory = deque(maxlen=memory_size)

# 定义策略网络
def build_policy_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_actions, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model

# 初始化策略网络
policy_model = build_policy_model()
target_model = build_policy_model()
target_model.set_weights(policy_model.get_weights())

# 选择特征提取方法
def select_feature_extraction_method(state):
    if np.random.rand() <= epsilon:
        return random.randint(0, n_actions - 1)  # 探索
    else:
        prob = policy_model.predict(state)
        return np.argmax(prob[0])  # 利用

# 奖励函数
def get_reward(recall, precision):
    if recall > 0.55:
        return precision
    else:
        return 0

# 存储经验
def store_memory(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# 训练策略网络
def train_model():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in batch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(target_model.predict(next_state)[0])
        target_f = policy_model.predict(state)
        target_f[0][action] = target
        policy_model.fit(state, target_f, epochs=1, verbose=0)

# 更新目标网络
def update_target_model():
    target_model.set_weights(policy_model.get_weights())

# 查询图像处理函数
def process_query_image(i, query_path, video_path, query_image, cache,query_num ):
    # 初始化状态

    state = cv2.resize(query_image, (64, 64))
    state = np.expand_dims(state, axis=0)  # 增加一个维度
    total_reward = []
    total_action = "start"  
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_flag = [1] * (total_frames + 2)

    for episode in range(10): 
        action = select_feature_extraction_method(state)
        done=0
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


        if flag==1:
            done=1 
            video_flag = [1] * (total_frames + 2)
            next_state = state
            reward=0
            store_memory(state, action, reward, next_state, done)
            state = next_state
            update_target_model()
            print(f'Total Reward: {total_reward}')
            print(f'Action Sequence: {total_action}')
            total_action = "start"
            total_reward = []
            continue
        
        reward = get_reward(recall[i], precision[i])
        total_reward.append(reward)
        
        next_state = state
        if reward == 0:
            done=1  # 假设flag为1表示找到匹配帧
        store_memory(state, action, reward, next_state, done)
        state = next_state
        
        if done==1:
            video_flag = [1] * (total_frames + 2)
            update_target_model()
            print(f'Total Reward: {total_reward}')
            print(f'Action Sequence: {total_action}')
            total_action = "start"
            total_reward = []
            continue

        train_model()

    return total_reward, total_action

# 主函数
def main():
    query_path = '/home/lzp/zby/query_taipei_test'
    video_path = '/home/lzp/zby/taipei_640.mp4'
    query_num = 20
    cap = cv2.VideoCapture(video_path)
    cap.release()
    query_inputs = glob.glob(os.path.join(query_path, "*"))
    cache={}
    with open('/home/lzp/zby/opencv/cache/taipei_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    for i in tqdm(range(query_num)):
        query_image = cv2.imread(query_inputs[i])
        if query_image is None:
            print("Error: Could not read query image.")
            return
    # 处理新查询图像
        total_reward, total_action = process_query_image(i, query_path, video_path,query_image ,cache,query_num=1 )

    # 保存模型
    policy_model.save('policy_model.h5')
    print("Model saved to file.")

if __name__ == "__main__":
    main()
