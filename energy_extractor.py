import os
import cv2
import numpy as np
import time
import glob
from skimage.feature import graycomatrix, graycoprops

from memory.memory_extractor import judge_memory, update_memory

def calculate_glcm(image, bins=256):
    # 转换为灰度图像
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算视频帧的灰度共生矩阵和纹理特征
    frame_glcm = graycomatrix(gray_frame, [1], [0], 256, symmetric=True, normed=True)
    frame_contrast = graycoprops(frame_glcm, 'contrast').flatten()
    frame_energy = graycoprops(frame_glcm, 'energy').flatten()
    return frame_contrast,frame_energy

def compare_similarity(vehicle_contrast, frame_contrast,vehicle_energy,frame_energy):
    """比较两个直方图"""
    contrast_similarity = np.linalg.norm(vehicle_contrast - frame_contrast)
    energy_similarity = np.linalg.norm(vehicle_energy - frame_energy)

    return contrast_similarity,energy_similarity
    
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
    contrasts=[]
    energys=[]

    # 假设 contours 是你通过 findContours 得到的轮廓列表
    # 假设 image 是原始图像
    for contour in contours:
        if cv2.contourArea(contour) > 800:  # 过滤掉小面积的轮廓
            x, y, w, h = cv2.boundingRect(contour)
        else:
            continue
        cropped_region = image[y:y+h, x:x+w]
        
        # 计算裁剪区域的纹理特征
        contrast,energy=calculate_glcm(cropped_region)
        contrasts.append(contrast)
        energys.append(energy)
        
    return contrasts,energys
def energy_extraction(query_path,video_path,query_num,video_flag, cache):
    # get background subtractor
    sub_type = 'MOG2' # 'MOG2'
    if sub_type == "MOG2":
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True,history=800)
        # backSub.setShadowThreshold(0.75)
    else:
        backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=800, detectShadows=True)

    folder_path=query_path
    query_inputs = glob.glob(folder_path + "/*")
    query_num=20
    frame_count=0
    video = video_path
    cap = cv2.VideoCapture(video)

    sum_count=0
    contrasts=[]
    energys=[]
    count=[]
    min=[]
    max=[]
    count = [0] * query_num
    for i in range(query_num):
        if query_path.endswith('.jpg'):
            img=cv2.imread(query_path)
        else:
            img = cv2.imread(query_inputs[i])
        contrast,energy=calculate_glcm(img)
        contrasts.append(contrast)
        energys.append(energy)
        query_frame=int(os.path.basename(query_inputs[i]).split('.')[0].split('_')[-1].split('-')[0])
        frame_len=int(os.path.basename(query_inputs[i]).split('.')[0].split('_')[-3])
        min.append(query_frame-frame_len//2)
        max.append(query_frame+frame_len//2)

    recall=[]
    recall = [0] * query_num
    while(1):
        ret, image = cap.read()
        if not ret:
            break
        # update the background model and obtain foreground mask
        frame_count=frame_count+1
        #判断是否需要提取
        if video_flag[frame_count]==0:
            continue

        save_flag,contrastss,energyss=judge_memory(1,frame_count,cache)
        if save_flag==0:
            fg_mask = backSub.apply(image)
            # 将掩码应用于原始图像以获得彩色前景
            contrastss,energyss=calculate(fg_mask,image)
            cache=update_memory(cache,1,frame_count,contrastss,energyss)
        
        # fg_mask = backSub.apply(image)
        #     # 将掩码应用于原始图像以获得彩色前景
        # contrastss,energyss=calculate(fg_mask,image)

        for i in range(query_num):
            # 比较直方图

            contrast_similarity_min = 1000
            energy_similarity_min = 1000
            for j in range(len(contrastss)):
                contrast_similarity_t,energy_similarity_t = compare_similarity(contrastss[j], contrasts[i],energyss[j],energys[i])
                if contrast_similarity_min>contrast_similarity_t and energy_similarity_min>energy_similarity_t:
                    contrast_similarity_min = contrast_similarity_t
                    energy_similarity_min = energy_similarity_t
            if contrast_similarity_min<50 and energy_similarity_min < 0.2:
                count[i]=count[i]+1
                if(frame_count>min[i] and frame_count<max[i]):
                    recall[i]=recall[i]+1
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
    energy_extraction()
