import os
import pickle
import cv2
import numpy as np
import time
import glob
from skimage.feature import local_binary_pattern

from memory.memory_extractor import judge_memory, update_memory

def calculate_lbp(image, P=8, R=1):
    """计算图像的LBP特征"""
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, P + 3), range=(0, P + 2))
    return hist / hist.sum()

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CHISQR):
    """比较两个直方图"""
    hist1 = hist1.astype('float32')
    hist2 = hist2.astype('float32')
    return cv2.compareHist(hist1, hist2, method)

def get_motion_mask(fg_mask, min_thresh=0, kernel=np.array((9,9), dtype=np.uint8)):
    """ Obtains image mask """
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

def calculate(fg_mask, image):
    motion_mask = get_motion_mask(fg_mask, kernel=np.array((9,9), dtype=np.uint8))
    color_foreground = cv2.bitwise_and(image, image, mask=motion_mask)
    color_foreground = cv2.cvtColor(color_foreground, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cumulative_histogram = np.zeros(10)
    hists = []

    for contour in contours:
        if cv2.contourArea(contour) > 700:  # 过滤掉小面积的轮廓
            x, y, w, h = cv2.boundingRect(contour)
        else:
            continue
        cropped_region = color_foreground[y:y+h, x:x+w]
        
        # 计算裁剪区域的LBP直方图
        histogram = calculate_lbp(cropped_region)
        hists.append(histogram)
        
        # 累加到综合直方图中
        cumulative_histogram += histogram

    return cumulative_histogram, hists

def lbp_extraction(query_path, video_path, query_num, video_flag, cache, start_frame, end_frame, index):
    # with open(cache_path, 'rb') as f:
    #     cache = pickle.load(f)
    sub_type = 'MOG2'
    if sub_type == "MOG2":
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True, history=800)
    else:
        backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=800, detectShadows=True)

    query_inputs = glob.glob(os.path.join(query_path, "*"))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)  # 设置开始帧
    frame_count = start_frame - 1

    # 加载指定的查询图像
    if query_path.endswith('.jpg'):
        img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(query_inputs[index], cv2.IMREAD_GRAYSCALE)
    query_hist = calculate_lbp(img)
    query_frame = int(os.path.basename(query_inputs[index]).split('.')[0].split('_')[-1].split('-')[0])
    frame_len = int(os.path.basename(query_inputs[index]).split('.')[0].split('_')[-3])
    min_frame = query_frame - frame_len // 2
    max_frame = query_frame + frame_len // 2

    # 初始化统计变量
    num_filtered_frames = 0
    num_kept_frames = 0
    num_true_positive_frames = 0
    num_detected_target_frames = 0

    while frame_count < end_frame:
        ret, image = cap.read()
        if not ret:
            break
        frame_count += 1

        # 判断是否需要提取
        if video_flag[frame_count] == 0:
            num_filtered_frames += 1
            continue
        else:
            num_kept_frames += 1

        # 判断当前帧是否应当保留（目标帧）
        if min_frame <= frame_count <= max_frame:
            num_true_positive_frames += 1

        # 检查缓存
        save_flag, cumulative_histogram, histss = judge_memory(2, frame_count, cache)
        if save_flag == 0:
            fg_mask = backSub.apply(image)
            cumulative_histogram, histss = calculate(fg_mask, image)
            cache = update_memory(cache, 2, frame_count, cumulative_histogram, histss)

        # 特征比较
        similarity_min = compare_histograms(cumulative_histogram, query_hist)
        for hist in histss:
            similarity_t = compare_histograms(hist, query_hist)
            if similarity_min > similarity_t:
                similarity_min = similarity_t

        if similarity_min < 0.6:
            # 匹配成功
            if min_frame <= frame_count <= max_frame:
                num_detected_target_frames += 1
        else:
            # 匹配失败，过滤帧
            video_flag[frame_count] = 0
            num_filtered_frames += 1

    cap.release()

    return num_filtered_frames, num_kept_frames, num_true_positive_frames, num_detected_target_frames, cache, video_flag

    
if __name__ == "__main__":
    lbp_extraction()
