import os
import cv2
import numpy as np
import glob
from skimage.feature import graycomatrix, graycoprops

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

def compare_similarity(vehicle_features, frame_features):
    contrast_similarity = np.linalg.norm(vehicle_features['contrast'] - frame_features['contrast'])
    energy_similarity = np.linalg.norm(vehicle_features['energy'] - frame_features['energy'])
    color_similarity = cv2.compareHist(vehicle_features['color_histogram'], frame_features['color_histogram'], cv2.HISTCMP_CORREL)
    return contrast_similarity, energy_similarity, color_similarity

def get_motion_mask(fg_mask, min_thresh=0, kernel=np.ones((9, 9), np.uint8)):
    _, thresh = cv2.threshold(fg_mask, min_thresh, 255, cv2.THRESH_BINARY)
    if thresh is not None and thresh.size > 0:
        motion_mask = cv2.medianBlur(thresh, 3)
    else:
        motion_mask = fg_mask
        return motion_mask
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return motion_mask

def extract_features(image):
    color_histogram = calculate_color_histogram(image)
    contrast, energy = calculate_glcm(image)
    return {
        'color_histogram': color_histogram,
        'contrast': contrast,
        'energy': energy
    }

def calculate(fg_mask, image):
    motion_mask = get_motion_mask(fg_mask, kernel=np.ones((9, 9), np.uint8))
    color_foreground = cv2.bitwise_and(image, image, mask=motion_mask)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features_list = []
    for contour in contours:
        if cv2.contourArea(contour) > 800:
            x, y, w, h = cv2.boundingRect(contour)
        else:
            continue
        cropped_region = image[y:y+h, x:x+w]
        features = extract_features(cropped_region)
        features_list.append(features)
    return features_list

def color_energy_extraction(query_path, video_path, query_num, video_flag, cache):
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True, history=800)
    query_inputs = glob.glob(os.path.join(query_path, "*"))
    frame_count = 0
    cap = cv2.VideoCapture(video_path)
    query_features = []
    min_frames = []
    max_frames = []

    for i in range(query_num):
        if query_path.endswith('.jpg'):
            image=cv2.imread(query_path)
        else:
            image = cv2.imread(query_inputs[i])
        features = extract_features(image)
        query_features.append(features)
        query_frame = int(os.path.basename(query_inputs[i]).split('.')[0].split('_')[-1].split('-')[0])
        frame_len = int(os.path.basename(query_inputs[i]).split('.')[0].split('_')[-3])
        min_frames.append(query_frame - frame_len // 2)
        max_frames.append(query_frame + frame_len // 2)

    count = [0] * query_num
    recall = [0] * query_num

    while True:
        ret, image = cap.read()
        if not ret:
            break
        frame_count += 1
        if video_flag[frame_count] == 0:
            continue

        save_flag,frame_features_list,n=judge_memory(4,frame_count,cache)
        if save_flag==0:
            fg_mask = backSub.apply(image)
            # 将掩码应用于原始图像以获得彩色前景
            frame_features_list = calculate(fg_mask, image)
            cache=update_memory(cache,4,frame_count,frame_features_list,0)

        # fg_mask = backSub.apply(image)
        # frame_features_list = calculate(fg_mask, image)

        for i in range(query_num):
            contrast_similarity_min = 1000
            energy_similarity_min = 1000
            color_similarity_min = 1000

            for frame_features in frame_features_list:
                contrast_similarity_t, energy_similarity_t, color_similarity_t = compare_similarity(query_features[i], frame_features)
                if contrast_similarity_min > contrast_similarity_t and energy_similarity_min > energy_similarity_t and color_similarity_min > color_similarity_t:
                    contrast_similarity_min = contrast_similarity_t
                    energy_similarity_min = energy_similarity_t
                    color_similarity_max = color_similarity_t
            
            if contrast_similarity_min < 60 and energy_similarity_min < 0.4 and color_similarity_max < 12:
                count[i] += 1
                if min_frames[i] < frame_count < max_frames[i]:
                    recall[i] += 1
            else:
                video_flag[frame_count] = 0
    print(count)
    re = []
    precision = []
    flag = 0
    for i in range(query_num):
        re.append(recall[i] / (max_frames[i] - min_frames[i]))
        if count[i] != 0:
            precision.append(recall[i] / count[i])
        else:
            flag = 1

    return re, precision, video_flag, flag, count, cache

