import os
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

def extract_color_lbp_features(image):
    color_histogram = calculate_color_histogram(image)
    lbp_histogram = calculate_lbp(image)
    return {
        'color_histogram': color_histogram,
        'lbp_histogram': lbp_histogram
    }

def compare_color_lbp(vehicle_features, frame_features):
    color_similarity = cv2.compareHist(vehicle_features['color_histogram'], frame_features['color_histogram'], cv2.HISTCMP_CORREL)
    lbp_similarity = compare_histograms(vehicle_features['lbp_histogram'], frame_features['lbp_histogram'])
    return color_similarity, lbp_similarity

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
def calculate_color_lbp(fg_mask, image):
    motion_mask = get_motion_mask(fg_mask)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features_list = []

    for contour in contours:
        if cv2.contourArea(contour) > 800:
            x, y, w, h = cv2.boundingRect(contour)
        else:
            continue
        cropped_region = image[y:y+h, x:x+w]
        features = extract_color_lbp_features(cropped_region)
        features_list.append(features)
    return features_list

def color_lbp_extraction(query_path, video_path, query_num, video_flag, cache):
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
        features = extract_color_lbp_features(image)
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

        save_flag,frame_features_list,n=judge_memory(3,frame_count,cache)
        if save_flag==0:
            fg_mask = backSub.apply(image)
            # 将掩码应用于原始图像以获得彩色前景
            frame_features_list = calculate_color_lbp(fg_mask, image)
            cache=update_memory(cache,3,frame_count,frame_features_list,0)
            
        # fg_mask = backSub.apply(image)
        # frame_features_list = calculate_color_lbp(fg_mask, image)

        for i in range(query_num):
            color_similarity_min = 1000
            lbp_similarity_min = 1000

            for frame_features in frame_features_list:
                color_similarity_t, lbp_similarity_t = compare_color_lbp(query_features[i], frame_features)
                if lbp_similarity_min > lbp_similarity_t and color_similarity_min > color_similarity_t:
                    lbp_similarity_min = lbp_similarity_t
                    color_similarity_max = color_similarity_t
            
            if color_similarity_min < 12 and lbp_similarity_min < 1:
                count[i] += 1
                if min_frames[i] < frame_count < max_frames[i]:
                    recall[i] += 1
            else:
                video_flag[frame_count] = 0

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


