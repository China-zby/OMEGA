import os
import pickle
import cv2
import numpy as np
import glob
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from memory.memory_extractor import judge_memory, update_memory

def calculate_glcm(image, bins=256):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_glcm = graycomatrix(gray_frame, [1], [0], 256, symmetric=True, normed=True)
    frame_contrast = graycoprops(frame_glcm, 'contrast').flatten()
    frame_energy = graycoprops(frame_glcm, 'energy').flatten()
    return frame_contrast, frame_energy

def calculate_color_histogram(image, bins=256):
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def calculate_lbp(image, P=8, R=1):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_frame, P, R, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, P + 3), range=(0, P + 2))
    return hist / hist.sum()

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CHISQR):
    hist1 = hist1.astype('float32')
    hist2 = hist2.astype('float32')
    return cv2.compareHist(hist1, hist2, method)
def extract_glcm_lbp_features(image):
    contrast, energy = calculate_glcm(image)
    lbp_histogram = calculate_lbp(image)
    return {
        'contrast': contrast,
        'energy': energy,
        'lbp_histogram': lbp_histogram
    }
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
def compare_glcm_lbp(vehicle_features, frame_features):
    contrast_similarity = np.linalg.norm(vehicle_features['contrast'] - frame_features['contrast'])
    energy_similarity = np.linalg.norm(vehicle_features['energy'] - frame_features['energy'])
    lbp_similarity = compare_histograms(vehicle_features['lbp_histogram'], frame_features['lbp_histogram'])
    return contrast_similarity, energy_similarity, lbp_similarity

def calculate_glcm_lbp(fg_mask, image):
    motion_mask = get_motion_mask(fg_mask)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features_list = []

    for contour in contours:
        if cv2.contourArea(contour) > 700:
            x, y, w, h = cv2.boundingRect(contour)
        else:
            continue
        cropped_region = image[y:y+h, x:x+w]
        features = extract_glcm_lbp_features(cropped_region)
        features_list.append(features)
    return features_list

def glcm_lbp_extraction(query_path, video_path, query_num, video_flag, cache, start_frame, end_frame, index):
    # with open(cache_path, 'rb') as f:
    #     cache = pickle.load(f)
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True, history=800)
    query_inputs = glob.glob(os.path.join(query_path, "*"))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    frame_count = start_frame - 1

    # 加载指定的查询图像
    if query_path.endswith('.jpg'):
        image = cv2.imread(query_path)
    else:
        image = cv2.imread(query_inputs[index])
    query_feature = extract_glcm_lbp_features(image)
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

        if video_flag[frame_count] == 0:
            num_filtered_frames += 1
            continue
        else:
            num_kept_frames += 1

        # 判断当前帧是否应当保留（目标帧）
        if min_frame <= frame_count <= max_frame:
            num_true_positive_frames += 1

        # 检查缓存
        save_flag, frame_features_list, n = judge_memory(5, frame_count, cache)
        if save_flag == 0:
            fg_mask = backSub.apply(image)
            frame_features_list = calculate_glcm_lbp(fg_mask, image)
            #cache = update_memory(cache, 5, frame_count, frame_features_list, 0)

        # 特征比较
        contrast_similarity_min = 1000
        energy_similarity_min = 1000
        lbp_similarity_min = 1000

        for frame_features in frame_features_list:
            contrast_similarity_t, energy_similarity_t, lbp_similarity_t = compare_glcm_lbp(query_feature, frame_features)
            if (contrast_similarity_t < contrast_similarity_min and
                energy_similarity_t < energy_similarity_min and
                lbp_similarity_t < lbp_similarity_min):
                contrast_similarity_min = contrast_similarity_t
                energy_similarity_min = energy_similarity_t
                lbp_similarity_min = lbp_similarity_t

        if contrast_similarity_min < 60 and energy_similarity_min < 0.4 and lbp_similarity_min < 0.7:
            # 匹配成功
            if min_frame <= frame_count <= max_frame:
                num_detected_target_frames += 1
        else:
            # 匹配失败，过滤帧
            video_flag[frame_count] = 0
            num_filtered_frames += 1

    cap.release()

    return num_filtered_frames, num_kept_frames, num_true_positive_frames, num_detected_target_frames, cache, video_flag
