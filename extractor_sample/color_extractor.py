import os
import pickle
import cv2
import numpy as np
import time
import glob

import os

from memory.memory_extractor import judge_memory
from memory.memory_extractor import update_memory

def calculate_histogram(image, bins=256):
    """计算图像的颜色直方图"""
    histogram = cv2.calcHist([image], [0], None, [bins], [0, 256])
    return cv2.normalize(histogram, histogram).flatten()

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CHISQR):
    """比较两个直方图"""
    hist1 = hist1.astype('float32')
    hist2 = hist2.astype('float32')
    return cv2.compareHist(hist1, hist2, method)

def get_motion_mask(fg_mask, min_thresh=0, kernel=np.array((9,9), dtype=np.uint8)):
    """ Obtains image mask
        Inputs: 
            fg_mask - foreground mask
            kernel - kernel for Morphological Operations
        Outputs: 
            mask - Thresholded mask for moving pixels
        """
    _, thresh = cv2.threshold(fg_mask,min_thresh,255,cv2.THRESH_BINARY)
    if thresh is not None and thresh.size > 0:
        motion_mask = cv2.medianBlur(thresh, 3)
    else:
        motion_mask=fg_mask
        return motion_mask

    
    # morphological operations
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return motion_mask

def calculate(fg_mask,image):
    motion_mask = get_motion_mask(fg_mask, kernel=np.array((9,9), dtype=np.uint8))
    color_foreground = cv2.bitwise_and(image, image, mask=motion_mask)
    color_foreground=cv2.cvtColor(color_foreground, cv2.COLOR_BGR2RGB)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化一个累加直方图
    cumulative_histogram = np.zeros(256)
    hists=[]

    # 假设 contours 是你通过 findContours 得到的轮廓列表
    # 假设 image 是原始图像
    for contour in contours:
        if cv2.contourArea(contour) > 700:  # 过滤掉小面积的轮廓
            x, y, w, h = cv2.boundingRect(contour)
        else:
            continue
        cropped_region = image[y:y+h, x:x+w]
        
        # 计算裁剪区域的颜色直方图
        histogram = calculate_histogram(cropped_region)
        hists.append(histogram)
        
        # 累加到综合直方图中
        cumulative_histogram += histogram
    return cumulative_histogram,hists
# def color_extraction(query_path, video_path, query_num, video_flag, cache, start_frame, end_frame, index):
#     # with open(cache_path, 'rb') as f:
#     #     cache = pickle.load(f)
#     # 初始化背景减法器
#     sub_type = 'MOG2'
#     if sub_type == "MOG2":
#         backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True, history=800)
#     else:
#         backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=800, detectShadows=True)
#     # 加载查询图像
#     s_time = time.time()
#     query_inputs = glob.glob(os.path.join(query_path, "*"))
#     cap = cv2.VideoCapture(video_path)
#     #print("video_read_time:",time.time()-s_time)
#     s_time = time.time()
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)  # 设置开始帧
#     frame_count = start_frame - 1
#     #print("video_set_time:",time.time()-s_time)
#     s_time = time.time()
#     # 只处理指定的 index
#     if query_path.endswith('.jpg'):
#         img = cv2.imread(query_path)
#     else:
#         img = cv2.imread(query_inputs[index])
#     query_hist = calculate_histogram(img)
#     query_frame = int(os.path.basename(query_inputs[index]).split('.')[0].split('_')[-1].split('-')[0])
#     frame_len = int(os.path.basename(query_inputs[index]).split('.')[0].split('_')[-3])
#     min_frame = query_frame - frame_len // 2
#     max_frame = query_frame + frame_len // 2
#     #print("img_read_time:",time.time()-s_time)
#     # 初始化统计变量
#     num_filtered_frames = 0          # 被过滤的帧数量
#     num_kept_frames = 0              # 保留的帧数量
#     num_true_positive_frames = 0     # 应当保留的目标帧数量
#     num_detected_target_frames = 0   # 成功保留的目标帧数量
#     total_io_time = 0
#     total_cache_time = 0
#     total_com_time = 0
#     while frame_count < end_frame:
#         s_time = time.time()
#         ret, image = cap.read()
#         if not ret:
#             break
#         frame_count += 1
#         total_io_time += time.time()-s_time
#         # 判断是否需要提取
#         if video_flag[frame_count] == 0:
#             num_filtered_frames += 1
#             continue
#         else:
#             num_kept_frames += 1

#         # 判断当前帧是否应当保留（目标帧）
#         if min_frame <= frame_count <= max_frame:
#             num_true_positive_frames += 1

#         s_time = time.time()
#         # 检查缓存
#         save_flag, cumulative_histogram, histss = judge_memory(0, frame_count, cache)
#         if save_flag == 0:
#             fg_mask = backSub.apply(image)
#             # 计算直方图
#             cumulative_histogram, histss = calculate(fg_mask, image)
            
#             cache = update_memory(cache, 0, frame_count, cumulative_histogram, histss)
#             total_cache_time += time.time()-s_time
#         s_time = time.time()
#         # 比较直方图
#         similarity_min = compare_histograms(cumulative_histogram, query_hist)
#         for hist in histss:
#             similarity_t = compare_histograms(hist, query_hist)
#             if similarity_min > similarity_t:
#                 similarity_min = similarity_t
#         total_com_time += time.time()-s_time
#         if similarity_min < 10:
#             # 匹配成功
#             if min_frame <= frame_count <= max_frame:
#                 num_detected_target_frames += 1
#         else:
#             # 匹配失败，过滤帧
#             video_flag[frame_count] = 0
#             num_filtered_frames += 1

#     cap.release()
#     #print("io:",total_io_time)
#     return num_filtered_frames, num_kept_frames, num_true_positive_frames, num_detected_target_frames, cache, video_flag

def color_extraction(query_path, video_path, query_num, video_flag, cache, cap, start_frame, end_frame, index, step=5):
    # 初始化背景减法器
    sub_type = 'MOG2'
    if sub_type == "MOG2":
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True, history=800)
    else:
        backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=800, detectShadows=True)
    
    # 加载查询图像
    query_inputs = glob.glob(os.path.join(query_path, "*"))
    # cap = cv2.VideoCapture(video_path)
    
    # # 定位到起始帧
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    
    # 当前处理的帧号
    current_frame = start_frame 

    if query_path.endswith('.jpg'):
        img = cv2.imread(query_path)
    else:
        img = cv2.imread(query_inputs[index])
    
    query_hist = calculate_histogram(img)
    
    # 解析 Query 图片对应的目标帧范围
    query_filename = os.path.basename(query_inputs[index])
    query_frame_center = int(query_filename.split('.')[0].split('_')[-1].split('-')[0])
    frame_len = int(query_filename.split('.')[0].split('_')[-3])
    min_frame = query_frame_center - frame_len // 2
    max_frame = query_frame_center + frame_len // 2

    # 初始化统计变量
    num_filtered_frames = 0
    num_kept_frames = 0
    num_true_positive_frames = 0
    num_detected_target_frames = 0

    # === 循环修改：不再逐帧，而是配合 step ===
    while current_frame <= end_frame: # 注意这里边界控制
        
        # 1. 读取当前这一帧 (Actual Read)
        ret, image = cap.read()
        if not ret:
            break
        
        # 计算当前这一步覆盖的范围 (例如：处理第100帧，step=5，则覆盖 100, 101, 102, 103, 104)
        # 注意不要超出 end_frame
        batch_end = min(current_frame + step, end_frame + 1)
        batch_size = batch_end - current_frame
        
        # 2. 检查当前帧是否已经被之前的步骤标记为过滤
        if video_flag[current_frame] == 0:
            # 如果当前帧已被过滤，则假设这一批次都过滤
            num_filtered_frames += batch_size
            
            # 物理跳过剩下的 step-1 帧，准备下一次循环
            for _ in range(step - 1):
                cap.grab() 
            
            current_frame += step
            continue

        # 3. 统计 Ground Truth (真实目标帧数量)
        # 检查这一批次里有多少帧属于目标范围
        # 范围求交集：[current_frame, batch_end) AND [min_frame, max_frame]
        overlap_start = max(current_frame, min_frame)
        overlap_end = min(batch_end - 1, max_frame) # batch_end是开区间，max_frame是闭区间，所以要小心
        
        # 简单的逐个检查 (为了准确性)
        current_batch_true_positives = 0
        for f_idx in range(current_frame, batch_end):
            if min_frame <= f_idx <= max_frame:
                current_batch_true_positives += 1
        
        num_true_positive_frames += current_batch_true_positives
        num_kept_frames += batch_size # 暂时假设保留，后面如果匹配失败再减去

        # 4. 特征提取与匹配逻辑 (只做一次)
        # 检查缓存
        save_flag, cumulative_histogram, histss = judge_memory(0, current_frame, cache)
        
        if save_flag == 0:
            fg_mask = backSub.apply(image)
            cumulative_histogram, histss = calculate(fg_mask, image)
            cache = update_memory(cache, 0, current_frame, cumulative_histogram, histss)
        
        similarity_min = compare_histograms(cumulative_histogram, query_hist)
        for hist in histss:
            similarity_t = compare_histograms(hist, query_hist)
            if similarity_min > similarity_t:
                similarity_min = similarity_t
        
        # 5. 决策与状态广播
        if similarity_min < 10:
            # === 匹配成功 (Keep) ===
            # 这一批次都被保留 (video_flag 默认为 1，不需要改)
            # 统计检测到的目标帧
            num_detected_target_frames += current_batch_true_positives
        else:
            # === 匹配失败 (Filter) ===
            # 将这一批次的所有帧标记为 0
            for f_idx in range(current_frame, batch_end):
                video_flag[f_idx] = 0
            
            # 修正统计数据
            num_filtered_frames += batch_size
            num_kept_frames -= batch_size # 刚才加了，现在要减回去

        # 6. 物理跳帧 (Fast Forward)
        # 我们已经 read() 了一帧，还需要跳过 step-1 帧才能到达下一个 current_frame
        # 使用 grab() 比 read() 快很多，因为它不解码图像
        frames_to_skip = step - 1
        for _ in range(frames_to_skip):
            if current_frame + 1 + _ < end_frame: # 边界检查
                cap.grab()
        
        current_frame += step

    #cap.release()
    return num_filtered_frames, num_kept_frames, num_true_positive_frames, num_detected_target_frames, cache, video_flag
    
if __name__ == "__main__":
    color_extraction()
