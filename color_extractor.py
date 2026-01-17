import os
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
        if cv2.contourArea(contour) > 800:  # 过滤掉小面积的轮廓
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
def color_extraction(query_path,video_path,query_num,video_flag, cache):
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
    sum_count=0
    cap = cv2.VideoCapture(video)
    hists=[]
    count=[]
    min=[]
    max=[]
    count = [0] * query_num
    for i in range(query_num):
        if query_path.endswith('.jpg'):
            img=cv2.imread(query_path)
        else:
            img = cv2.imread(query_inputs[i])
        hists.append(calculate_histogram(img))
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
        
        save_flag,cumulative_histogram,histss=judge_memory(0,frame_count,cache)
        if save_flag==0:
            fg_mask = backSub.apply(image)
            # 将掩码应用于原始图像以获得彩色前景
            cumulative_histogram,histss=calculate(fg_mask,image)
            cache=update_memory(cache,0,frame_count,cumulative_histogram,histss)

        # fg_mask = backSub.apply(image)
        # # 将掩码应用于原始图像以获得彩色前景
        # cumulative_histogram,histss=calculate(fg_mask,image)

        for i in range(query_num):
            # 比较直方图
            similarity = compare_histograms(cumulative_histogram, hists[i])
            similarity_min = similarity
            for j in range(len(histss)):
                similarity_t = compare_histograms(histss[j], hists[i])
                if similarity_min>similarity_t:
                    similarity_min = similarity_t
            if similarity_min<10:
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
    color_extraction()
