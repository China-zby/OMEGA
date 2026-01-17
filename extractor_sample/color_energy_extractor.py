import os
import pickle
import cv2
import numpy as np
import glob
from skimage.feature import graycomatrix, graycoprops

from memory.memory_extractor import judge_memory, update_memory

# def calculate_glcm(image, bins=256):
#     gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     frame_glcm = graycomatrix(gray_frame, [1], [0], 256, symmetric=True, normed=True)
#     frame_contrast = graycoprops(frame_glcm, 'contrast').flatten()
#     frame_energy = graycoprops(frame_glcm, 'energy').flatten()
#     return frame_contrast, frame_energy

# def calculate_color_histogram(image, bins=256):
#     hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
#     cv2.normalize(hist, hist)
#     return hist.flatten()

# def compare_similarity(vehicle_features, frame_features):
#     contrast_similarity = np.linalg.norm(vehicle_features['contrast'] - frame_features['contrast'])
#     energy_similarity = np.linalg.norm(vehicle_features['energy'] - frame_features['energy'])
#     color_similarity = cv2.compareHist(vehicle_features['color_histogram'], frame_features['color_histogram'], cv2.HISTCMP_CORREL)
#     return contrast_similarity, energy_similarity, color_similarity

# def get_motion_mask(fg_mask, min_thresh=0, kernel=np.array((9,9), dtype=np.uint8)):
#     """ Obtains image mask
#         Inputs: 
#             fg_mask - foreground mask
#             kernel - kernel for Morphological Operations
#         Outputs: 
#             mask - Thresholded mask for moving pixels
#         """
#     _, thresh = cv2.threshold(fg_mask,min_thresh,255,cv2.THRESH_BINARY)
#     if thresh is not None and thresh.size > 0:
#         motion_mask = cv2.medianBlur(thresh, 3)
#     else:
#         motion_mask=fg_mask
#         return motion_mask

    
#     # morphological operations
#     motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

#     return motion_mask

# def extract_features(image):
#     color_histogram = calculate_color_histogram(image)
#     contrast, energy = calculate_glcm(image)
#     return {
#         'color_histogram': color_histogram,
#         'contrast': contrast,
#         'energy': energy
#     }

# def calculate(fg_mask, image):
#     motion_mask = get_motion_mask(fg_mask, kernel=np.ones((9, 9), np.uint8))
#     color_foreground = cv2.bitwise_and(image, image, mask=motion_mask)
#     contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     features_list = []
#     for contour in contours:
#         if cv2.contourArea(contour) > 800:
#             x, y, w, h = cv2.boundingRect(contour)
#         else:
#             continue
#         cropped_region = image[y:y+h, x:x+w]
#         features = extract_features(cropped_region)
#         features_list.append(features)
#     return features_list

def calculate_histogram_color(image, bins=256):
    """计算图像的颜色直方图"""
    histogram = cv2.calcHist([image], [0], None, [bins], [0, 256])
    return cv2.normalize(histogram, histogram).flatten()

def compare_histograms_color(hist1, hist2, method=cv2.HISTCMP_CHISQR):
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

def calculate_color(fg_mask,image):
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
        histogram = calculate_histogram_color(cropped_region)
        hists.append(histogram)
        
        # 累加到综合直方图中
        cumulative_histogram += histogram
    return cumulative_histogram,hists

def calculate_glcm(image, bins=256):
    # 转换为灰度图像
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算视频帧的灰度共生矩阵和纹理特征
    frame_glcm = graycomatrix(gray_frame, [1], [0], 256, symmetric=True, normed=True)
    frame_contrast = graycoprops(frame_glcm, 'contrast').flatten()
    frame_energy = graycoprops(frame_glcm, 'energy').flatten()
    return frame_contrast,frame_energy

def compare_similarity_energy(vehicle_contrast, frame_contrast,vehicle_energy,frame_energy):
    """比较两个直方图"""
    contrast_similarity = np.linalg.norm(vehicle_contrast - frame_contrast)
    energy_similarity = np.linalg.norm(vehicle_energy - frame_energy)

    return contrast_similarity,energy_similarity

def calculate_energy(fg_mask,image):
    motion_mask = get_motion_mask(fg_mask, kernel=np.array((9,9), dtype=np.uint8))
    color_foreground = cv2.bitwise_and(image, image, mask=motion_mask)
    color_foreground=cv2.cvtColor(color_foreground, cv2.COLOR_BGR2RGB)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化一个累加直方图
    cumulative_histogram = np.zeros(256)
    contrasts=[]
    energys=[]

    # 假设 contours 是你通过 findContours 得到的轮廓列表
    # 假设 image 是原始图像
    for contour in contours:
        if cv2.contourArea(contour) > 700:  # 过滤掉小面积的轮廓
            x, y, w, h = cv2.boundingRect(contour)
        else:
            continue
        cropped_region = image[y:y+h, x:x+w]
        
        # 计算裁剪区域的纹理特征
        contrast,energy=calculate_glcm(cropped_region)
        contrasts.append(contrast)
        energys.append(energy)
        
    return contrasts,energys

def color_energy_extraction(query_path, video_path, query_num, video_flag, cache, start_frame, end_frame, index):
    # with open(cache_path, 'rb') as f:
    #     cache = pickle.load(f)

    # backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True, history=800)
    # query_inputs = glob.glob(os.path.join(query_path, "*"))
    # cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    # frame_count = start_frame - 1

    # # 加载指定查询图像
    # if query_path.endswith('.jpg'):
    #     image = cv2.imread(query_path)
    # else:
    #     image = cv2.imread(query_inputs[index])
    # query_feature = extract_features(image)
    # query_frame = int(os.path.basename(query_inputs[index]).split('.')[0].split('_')[-1].split('-')[0])
    # frame_len = int(os.path.basename(query_inputs[index]).split('.')[0].split('_')[-3])
    # min_frame = query_frame - frame_len // 2
    # max_frame = query_frame + frame_len // 2

    # # 初始化统计变量
    # num_filtered_frames = 0
    # num_kept_frames = 0
    # num_true_positive_frames = 0
    # num_detected_target_frames = 0

    # while frame_count < end_frame:
    #     ret, image = cap.read()
    #     if not ret:
    #         break
    #     frame_count += 1

    #     if video_flag[frame_count] == 0:
    #         num_filtered_frames += 1
    #         continue
    #     else:
    #         num_kept_frames += 1

    #     # 判断当前帧是否应当保留（目标帧）
    #     if min_frame <= frame_count <= max_frame:
    #         num_true_positive_frames += 1

    #     # 检查缓存
    #     save_flag, frame_features_list, n = judge_memory(4, frame_count, cache)
    #     # save_flag = 0
    #     if save_flag == 0:
    #         fg_mask = backSub.apply(image)
    #         frame_features_list = calculate(fg_mask, image)
    #         #print(frame_features_list)
    #         #cache = update_memory(cache, 4, frame_count, frame_features_list, 0)

    #     # 特征比较
    #     contrast_similarity_min = 1000
    #     energy_similarity_min = 1000
    #     color_similarity_min = 1000

    #     for frame_features in frame_features_list:
    #         contrast_similarity_t, energy_similarity_t, color_similarity_t = compare_similarity(query_feature, frame_features)
    #         if (contrast_similarity_t < contrast_similarity_min and
    #             energy_similarity_t < energy_similarity_min and
    #             color_similarity_t < color_similarity_min):
    #             contrast_similarity_min = contrast_similarity_t
    #             energy_similarity_min = energy_similarity_t
    #             color_similarity_min = color_similarity_t

    #     if contrast_similarity_min < 60 and energy_similarity_min < 0.4 and color_similarity_min < 12:
    #         if min_frame <= frame_count <= max_frame:
    #             num_detected_target_frames += 1
    #     else:
    #         video_flag[frame_count] = 0
    #         num_filtered_frames += 1

    # cap.release()

    # return num_filtered_frames, num_kept_frames, num_true_positive_frames, num_detected_target_frames, cache, video_flag
    sub_type = 'MOG2'
    if sub_type == "MOG2":
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True, history=800)
    else:
        backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=800, detectShadows=True)

    # 加载查询图像
    query_inputs = glob.glob(os.path.join(query_path, "*"))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)  # 设置开始帧
    frame_count = start_frame - 1

    # 只处理指定的 index
    if query_path.endswith('.jpg'):
        img = cv2.imread(query_path)
    else:
        img = cv2.imread(query_inputs[index])
    query_hist = calculate_histogram_color(img)
    query_frame = int(os.path.basename(query_inputs[index]).split('.')[0].split('_')[-1].split('-')[0])
    frame_len = int(os.path.basename(query_inputs[index]).split('.')[0].split('_')[-3])
    min_frame = query_frame - frame_len // 2
    max_frame = query_frame + frame_len // 2

    # 初始化统计变量
    num_filtered_frames = 0          # 被过滤的帧数量
    num_kept_frames = 0              # 保留的帧数量
    num_true_positive_frames = 0     # 应当保留的目标帧数量
    num_detected_target_frames = 0   # 成功保留的目标帧数量

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
        save_flag, cumulative_histogram, histss = judge_memory(0, frame_count, cache)
        if save_flag == 0:
            fg_mask = backSub.apply(image)
            # 计算直方图
            cumulative_histogram, histss = calculate_color(fg_mask, image)
            
            cache = update_memory(cache, 0, frame_count, cumulative_histogram, histss)

        # 比较直方图
        similarity_min = compare_histograms_color(cumulative_histogram, query_hist)
        for hist in histss:
            similarity_t = compare_histograms_color(hist, query_hist)
            if similarity_min > similarity_t:
                similarity_min = similarity_t

        if similarity_min < 12:
            # 匹配成功
            if min_frame <= frame_count <= max_frame:
                
                contrast_query, energy_query = calculate_glcm(img)
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
                    save_flag, contrast_list, energy_list = judge_memory(1, frame_count, cache)
                    if save_flag == 0:
                        fg_mask = backSub.apply(image)
                        contrast_list, energy_list = calculate_energy(fg_mask, image)
                        cache = update_memory(cache, 1, frame_count, contrast_list, energy_list)

                    # 特征比较
                    contrast_similarity_min = 1000
                    energy_similarity_min = 1000

                    for contrast_frame, energy_frame in zip(contrast_list, energy_list):
                        contrast_similarity_t, energy_similarity_t = compare_similarity_energy(contrast_frame, contrast_query, energy_frame, energy_query)
                        if contrast_similarity_t < contrast_similarity_min and energy_similarity_t < energy_similarity_min:
                            contrast_similarity_min = contrast_similarity_t
                            energy_similarity_min = energy_similarity_t

                    if contrast_similarity_min < 60 and energy_similarity_min < 0.4:
                        # 匹配成功
                        if min_frame <= frame_count <= max_frame:
                            num_detected_target_frames += 1
                        else:
                            # 匹配失败，过滤帧
                            video_flag[frame_count] = 0
                            num_filtered_frames += 1
        else:
            # 匹配失败，过滤帧
            video_flag[frame_count] = 0
            num_filtered_frames += 1

    cap.release()

    return num_filtered_frames, num_kept_frames, num_true_positive_frames, num_detected_target_frames, cache, video_flag
