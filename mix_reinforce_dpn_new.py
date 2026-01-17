from concurrent.futures import ThreadPoolExecutor
import glob
import multiprocessing as mp
from multiprocessing import Pool
import os
import time
import cv2
import numpy as np
import random
import tensorflow as tf
from functools import partial
from joblib import dump, load
import argparse
import joblib

from tensorflow.keras import models, layers, optimizers
from collections import deque
import pickle
from tqdm import trange

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
sys.path.append(os.path.join(os.path.dirname(__file__), 'memory'))
from tqdm import tqdm
from extractor_sample.color_extractor import color_extraction
from extractor_sample.energy_extractor import energy_extraction
from extractor_sample.lbp_extractor import lbp_extraction
from extractor_sample.net_extractor import net_extraction
from extractor_sample.color_energy_extractor import color_energy_extraction
from extractor_sample.color_lbp_extractor import color_lbp_extraction
from extractor_sample.energy_lbp_extractor import glcm_lbp_extraction

tf.get_logger().setLevel('ERROR')
# 定义参数
n_actions = 7  # 行动空间大小（7种特征提取方法）
epsilon = 0.5  # 探索概率
epsilon_min = 0.1
epsilon_decay = 0.995
gamma = 0.9  # 折扣因子
learning_rate = 0.001
batch_size = 8
memory_size = 200  # 经验回放缓冲区大小

# 初始化经验回放缓冲区
memory = deque(maxlen=memory_size)

# 定义策略网络
def build_policy_model(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_actions, activation='linear'))  # 输出每个动作的Q值
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model

# 选择特征提取方法
def select_feature_extraction_method(state):
    if np.random.rand() <= epsilon:
        return random.randint(0, n_actions - 1)  # 探索
    else:
        q_values = policy_model.predict(state, verbose=0)
        return np.argmax(q_values[0])  # 利用

# 奖励函数
def get_reward(filtering_efficiency, target_frame_retention_rate):
    if target_frame_retention_rate == 1.0:
        return filtering_efficiency
    elif target_frame_retention_rate < 0.5:
        penalty = 5.0
        return -penalty
    else:
        return filtering_efficiency*target_frame_retention_rate

# 存储经验
def store_memory(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# 训练策略网络
def train_model(pbar):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states = np.array([sample[0][0] for sample in batch])
    actions = np.array([sample[1] for sample in batch])
    rewards = np.array([sample[2] for sample in batch])
    next_states = np.array([sample[3][0] for sample in batch])
    dones = np.array([sample[4] for sample in batch])

    # 计算当前状态的Q值
    q_values = policy_model.predict(states, verbose=0)
    # 计算下一个状态的最大Q值
    q_next = target_model.predict(next_states, verbose=0)
    # 更新Q值
    for i in range(batch_size):
        target = q_values[i]
        if dones[i]:
            target[actions[i]] = rewards[i]
        else:
            target[actions[i]] = rewards[i] + gamma * np.amax(q_next[i])

    # 训练模型并获取损失值
    history = policy_model.fit(states, q_values, epochs=1, verbose=0) #2025.5.6修改epoch100->1
    loss = history.history['loss'][-1]  # 获取最后一个epoch的损失值
    
    # 更新进度条信息
    pbar.set_postfix(loss=loss)

    del states, actions, rewards, next_states, dones, q_values, q_next, target

# 更新目标网络
def update_target_model():
    #print("update_target_model...")
    target_model.set_weights(policy_model.get_weights())

def perform_extraction(action, query_path, video_path, query_num, video_flag, cache, cap, start_frame, end_frame, index):
    if action == 0:
        #print("Net Extraction")
        return net_extraction(
            query_path, video_path, query_num, video_flag, cache, start_frame, end_frame, index)
    elif action == 1:
        #print("Energy Extraction")
        return energy_extraction(
            query_path, video_path, query_num, video_flag, cache, start_frame, end_frame, index)
    elif action == 2:
        #print("LBP Extraction")
        return lbp_extraction(
            query_path, video_path, query_num, video_flag, cache, start_frame, end_frame, index)
    elif action == 3:
        #print("Color LBP Extraction")
        return color_lbp_extraction(
            query_path, video_path, query_num, video_flag, cache, start_frame, end_frame, index)
    elif action == 4:
        #print("Color Energy Extraction")
        return color_energy_extraction(
            query_path, video_path, query_num, video_flag, cache, start_frame, end_frame, index)
    elif action == 5:
        #print("GLCM LBP Extraction")
        return glcm_lbp_extraction(
            query_path, video_path, query_num, video_flag, cache, start_frame, end_frame, index)
    else:
        #print("Color Extraction")
        return color_extraction(
            query_path, video_path, query_num, video_flag, cache, cap, start_frame, end_frame, index, step=5)

# 构建状态向量
def construct_state(current_performance, previous_methods, previous_performance):
    # current_performance: [filtering_efficiency, target_frame_retention_rate]
    # previous_performance: [avg_filtering_efficiency, avg_target_frame_retention_rate]

    # 将 previous_methods 转换为 One-Hot 编码
    method_one_hot = np.zeros(n_actions)
    if len(previous_methods) > 0:
        method_one_hot[previous_methods[-1]] = 1
    else:
        method_one_hot = np.zeros(n_actions)

    # 构建状态向量
    state = np.concatenate([
        current_performance,    # 当前性能指标
        method_one_hot,         # 最近一次选择的特征提取方法的 One-Hot 编码
        previous_performance    # 之前阶段的平均性能指标
    ])
    state = np.expand_dims(state, axis=0)
    return state

# 查询图像处理函数
def process_query_image(i, query_path, video_path, cache, query_num, num_phases, pbar, phase_length):

    # if os.path.exists(cache_path):
    #     cache = load(cache_path)
    # if os.path.exists(cache_path):
    #     with open(cache_path, 'rb') as f:
    #         cache = pickle.load(f)

    #print(cache[1566])
    total_reward = []
    total_action = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # cap.release()

    #phase_length = 50
    num_phases = total_frames // phase_length + 1

    previous_methods = []         
    previous_performance = []     
    video_flag = [1] * (total_frames + 2)
    for phase in range(num_phases):
        if phase == 0:
            current_performance = [0, 0]  
            avg_performance = [0, 0]
        else:
            avg_filtering_efficiency = np.mean([p[0] for p in previous_performance])
            avg_target_frame_retention_rate = np.mean([p[1] for p in previous_performance])
            avg_performance = [avg_filtering_efficiency, avg_target_frame_retention_rate]

        state = construct_state(current_performance, previous_methods, avg_performance)

        while True:
            action = select_feature_extraction_method(state)
            if action in [3]:
                break
        total_action.append(action)

        start_frame = phase * phase_length
        end_frame = min((phase + 1) * phase_length, total_frames)
        
        num_filtered_frames, num_kept_frames, num_true_positive_frames, num_detected_target_frames, cache, video_flag = perform_extraction(
            action, query_path, video_path, query_num, video_flag, cache, cap, start_frame, end_frame, i)
        
        total_frames_in_phase = end_frame - start_frame
        effective_frames = total_frames_in_phase - num_true_positive_frames
        filtering_efficiency = num_filtered_frames / (effective_frames + 1e-6)

        if num_true_positive_frames > 0:
            target_frame_retention_rate = num_detected_target_frames / num_true_positive_frames
        else:
            target_frame_retention_rate = 1.0

        current_performance = [filtering_efficiency, target_frame_retention_rate]

        reward = get_reward(filtering_efficiency, target_frame_retention_rate)
        total_reward.append(reward)

        done = (phase == num_phases - 1)

        next_state = construct_state(current_performance, previous_methods + [action], avg_performance)

        store_memory(state, action, reward, next_state, done)

        train_model(pbar)  # 传递进度条以显示损失
        update_target_model()

        global epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        previous_methods.append(action)
        previous_performance.append(current_performance)

        pbar.update(1)

        if done:
            break
    video_flags.append(video_flag)
    # with open(cache_path, 'wb') as f:
    # # 使用 pickle.dump() 将对象写入文件
    #     pickle.dump(cache, f)
    

    del video_flag

    
    # 强制进行垃圾回收
    import gc
    gc.collect()
    return total_reward, total_action

# 包装函数（移除pbar参数）
def process_query_image_wrapper(i, query_path, video_path, cache_path, query_num, num_phases, phase_length):
    # 原函数逻辑，移除pbar相关操作
    process_query_image(i, query_path, video_path, cache_path, query_num, num_phases, None, phase_length)



# 主函数
def main():
    DEBUG = False
    parser = argparse.ArgumentParser(description="Set parameters for the training process.")
    parser.add_argument('--input_epoch', type=int, default=1, help="Number of input epochs")
    parser.add_argument('--query_path', type=str, default='/home/lzp/zby/query_taipei_test', help="query_path")
    parser.add_argument('--video_path', type=str, default='/home/lzp/zby/taipei_640.mp4', help="video_path")
    parser.add_argument('--cache_path', type=str, default='/home/lzp/zby/opencv/cache/taipei_cache_shell.pkl', help="cache_path")
    parser.add_argument('--query_num', type=int, default=10, help="Number of query")
    parser.add_argument('--phase_length', type=int, default=100, help="phase_length")

    # Parse the arguments
    args = parser.parse_args()
    
    query_path = args.query_path
    video_path = args.video_path
    query_num = args.query_num
    cache_path = args.cache_path
    n_epochs = 1  # 设置总训练轮次
    phase_length = args.phase_length
    input_epoch = args.input_epoch
    camera_id = os.path.splitext(os.path.basename(video_path))[0]
    out_flag = {}
    # 构建状态向量的维度
    state_dim = 2 + n_actions + 2  # [current_performance] + [method_one_hot] + [previous_performance]
    global policy_model, target_model, video_flags
    video_flags = []
    model_path = f'/home/lzp/zby/opencv/checkpoint/policy_model_{input_epoch}epochs_{phase_length}length.h5'
    if os.path.exists(model_path):
        policy_model = tf.keras.models.load_model(model_path)
        print(f"success load policy_model_{input_epoch}epochs_{phase_length}length.h5")
    else:
        policy_model = build_policy_model(state_dim)
    target_model = build_policy_model(state_dim)
    target_model.set_weights(policy_model.get_weights())
    if os.path.exists(cache_path):
        try:
            # joblib 读取大文件比 pickle 更稳
            cache = joblib.load(cache_path)
        except Exception as e:
            print(f"缓存文件损坏，将重新生成: {e}")
            cache = {}
    # 获取视频的总帧数
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        # 获取视频的总帧数
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 定义阶段的长度，例如每1000帧一个阶段
        num_phases = total_frames // phase_length + 1
        total_iterations = query_num * num_phases
        s_time = time.time()
        with tqdm(total=total_iterations, desc=f'\rEpoch {epoch + 1} - Processing Query Images and Phases') as pbar:
            for i in range(query_num):
                s_time = time.time()    
                process_query_image(i, query_path, video_path, cache, query_num, num_phases, pbar, phase_length)
                print("total time:",time.time()-s_time)
            joblib.dump(cache, cache_path, compress=3)
            # 在循环处替换为以下代码
        # process_func = partial(
        #     process_query_image_wrapper,
        #     query_path=query_path,
        #     video_path=video_path,
        #     cache_path=cache_path,
        #     query_num=query_num,
        #     num_phases=num_phases,
        #     phase_length=phase_length
        # )
        # mp.set_start_method("spawn")  # 在程序入口设置
        # with tqdm(total=query_num, desc=f'Epoch {epoch + 1} - Processing Query Images and Phases') as pbar:
        #     with Pool(processes=4) as pool:  # 进程数根据CPU核心数调整
        #         for _ in pool.imap(process_func, range(query_num)):
        #             pbar.update(1)

        # 减少探索率（epsilon），逐渐让模型更多地利用当前策略
        global epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # 更新目标网络的权重
        update_target_model()

        print(f"Completed epoch {epoch + 1}/{n_epochs}")
        if DEBUG == False:
            if (epoch+1)%1 == 0: 
                policy_model.save(f'/home/lzp/zby/opencv/checkpoint/policy_model_{input_epoch+1}epochs_{phase_length}length.h5')
                print("model saved to file.")
        else:
            print("debug done!")
    if DEBUG == False:
        with open(f'/home/lzp/zby/opencv/output/video_flags_{input_epoch}epochs_{camera_id}.txt', 'w') as f:
            for flag in video_flags:
                f.write(f"{flag}\n")  # 每个 flag 写入一行
    # 保存最终模型
    # policy_model.save(f'/home/lzp/zby/opencv/checkpoint/policy_model_{epoch+21}epochs_{phase_length}length.h5')
    # print("Final model saved to file.")

if __name__ == "__main__":
    main()