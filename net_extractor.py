import os
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
        if cv2.contourArea(contour) > 800:  # 过滤掉小面积的轮廓
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

def net_extraction(query_path,video_path,query_num,video_flag, cache):
    sub_type = 'MOG2' # 'MOG2'
    if sub_type == "MOG2":
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True, history=800)
    else:
        backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=800, detectShadows=True)

    folder_path=query_path
    query_inputs = glob.glob(folder_path + "/*")
    query_num=20
    frame_count=0
    video = video_path

    cap = cv2.VideoCapture(video)

    sum_count = 0
    features_list = []
    count = []
    min = []
    max = []
    count = [0] * query_num
    max_len = 1280  # 设定特征向量长度
    for i in range(query_num):
        if query_path.endswith('.jpg'):
            img=cv2.imread(query_path)
        else:
            img = cv2.imread(query_inputs[i])
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        features = extract_features(input_tensor, model)
        if features.size < max_len:
            features = np.pad(features, (0, max_len - features.size), 'constant')
        else:
            features = features[:max_len]
        features_list.append(features)
        query_frame = int(os.path.basename(query_inputs[i]).split('.')[0].split('_')[-1].split('-')[0])
        frame_len = int(os.path.basename(query_inputs[i]).split('.')[0].split('_')[-3])
        min.append(query_frame - frame_len // 2)
        max.append(query_frame + frame_len // 2)

    recall = [0] * query_num

    while True:
        start = time.time()
        ret, image = cap.read()
        if not ret:
            break
        frame_count += 1
        #判断是否需要提取
        if video_flag[frame_count]==0:
            continue

        save_flag, cumulative_features, features_lists=judge_memory(6,frame_count,cache)
        if save_flag==0:
            fg_mask = backSub.apply(image)
            # 将掩码应用于原始图像以获得彩色前景
            cumulative_features, features_lists=calculate(fg_mask,image)
            cache=update_memory(cache,6,frame_count,cumulative_features, features_lists)
            
        # fg_mask = backSub.apply(image)
        #     # 将掩码应用于原始图像以获得彩色前景
        # cumulative_features, features_lists=calculate(fg_mask,image)
        for i in range(query_num):
            similarity = compare_features(cumulative_features, features_list[i])
            similarity_min = similarity
            for j in range(len(features_lists)):
                similarity_t = compare_features(features_lists[j], features_list[i])
                if similarity_min > similarity_t:
                    similarity_min = similarity_t
            if similarity_min < 10:
                count[i] += 1
                if min[i] < frame_count < max[i]:
                    recall[i] += 1
            else:
                video_flag[frame_count]=0
    print(count)
    re=[]
    presion=[]
    flag=0
    for i in range(query_num):
        re.append(recall[i]/(max[i]-min[i]))
        if count[i]!=0:
            presion.append(recall[i]/count[i])
        else:
            flag=1

    return re,presion,video_flag,flag,count, cache
    
if __name__ == "__main__":
    net_extraction()
