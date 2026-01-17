import os
import pickle
import cv2
import numpy as np
import time
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image

from memory.memory_extractor import judge_memory, update_memory

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练的MobileNetV3-Small模型并移到GPU
model = mobilenet_v3_small(weights='IMAGENET1K_V1').to(device)
model.eval()  # 设置模型为评估模式

# 移除最后的分类层
model = torch.nn.Sequential(*list(model.children())[:-1])

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_images(image_paths):
    input_tensors = []
    for image_path in image_paths:
        img = Image.open(image_path)
        input_tensor = preprocess(img)
        input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
        input_tensors.append(input_tensor)
    input_tensors = torch.cat(input_tensors, dim=0)  # 合并成一个批次
    input_tensors = input_tensors.to(device)  # 将输入张量移到GPU
    return input_tensors

def extract_features(input_tensor, model):
    """提取特征向量并确保长度一致"""
    with torch.no_grad():
        features = model(input_tensor)
    features = features.squeeze().cpu().numpy()
    return features

def compare_features(feat1, feat2, method='euclidean'):
    """比较两个特征向量"""
    if method == 'euclidean':
        return np.linalg.norm(feat1 - feat2)
    else:
        raise ValueError("Unsupported comparison method")

def get_motion_mask(fg_mask, min_thresh=0, kernel=np.array((9,9), dtype=np.uint8)):
    """Obtains image mask"""
    _, thresh = cv2.threshold(fg_mask, min_thresh, 255, cv2.THRESH_BINARY)
    if thresh is not None and thresh.size > 0:
        motion_mask = cv2.medianBlur(thresh, 3)
    else:
        motion_mask = fg_mask
        return motion_mask
    
    # morphological operations
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return motion_mask

def calculate(fg_mask, image):
    motion_mask = get_motion_mask(fg_mask, kernel=np.array((9,9), dtype=np.uint8))
    color_foreground = cv2.bitwise_and(image, image, mask=motion_mask)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_len = 1280  # 设定特征向量长度
    cumulative_features = np.zeros(max_len)
    features_list = []

    for contour in contours:
        if cv2.contourArea(contour) > 700:  # 过滤掉小面积的轮廓
            x, y, w, h = cv2.boundingRect(contour)
        else:
            continue
        cropped_region = image[y:y+h, x:x+w]
        img_pil = Image.fromarray(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

        # 计算裁剪区域的特征
        features = extract_features(input_tensor, model)
        if features.size < max_len:
            features = np.pad(features, (0, max_len - features.size), 'constant')
        else:
            features = features[:max_len]
        features_list.append(features)
        
        # 累加到综合特征中
        cumulative_features += features

    return cumulative_features, features_list

def net_extraction(query_path, video_path, query_num, video_flag, cache, start_frame, end_frame, index):
    # with open(cache_path, 'rb') as f:
    #     cache = pickle.load(f)
    sub_type = 'MOG2'
    if sub_type == "MOG2":
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True, history=800)
    else:
        backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=800, detectShadows=True)

    query_inputs = glob.glob(os.path.join(query_path, "*"))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    frame_count = start_frame - 1

    # 加载指定的查询图像
    if query_path.endswith('.jpg'):
        img = cv2.imread(query_path)
    else:
        img = cv2.imread(query_inputs[index])
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
    query_feature = extract_features(input_tensor, model)
    max_len = 1280
    if query_feature.size < max_len:
        query_feature = np.pad(query_feature, (0, max_len - query_feature.size), 'constant')
    else:
        query_feature = query_feature[:max_len]
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
        save_flag, cumulative_features, features_lists = judge_memory(6, frame_count, cache)
        if save_flag == 0:
            fg_mask = backSub.apply(image)
            cumulative_features, features_lists = calculate(fg_mask, image)
            cache = update_memory(cache, 6, frame_count, cumulative_features, features_lists)

        # 特征比较
        similarity_min = compare_features(cumulative_features, query_feature)
        for features in features_lists:
            similarity_t = compare_features(features, query_feature)
            if similarity_min > similarity_t:
                similarity_min = similarity_t

        if similarity_min < 10:
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
    net_extraction()
