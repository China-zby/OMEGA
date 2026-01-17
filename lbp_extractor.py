import os
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
        if cv2.contourArea(contour) > 800:  # 过滤掉小面积的轮廓
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

def lbp_extraction(query_path,video_path,query_num,video_flag, cache):
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
    hists = []
    count = []
    min = []
    max = []
    count = [0] * query_num
    for i in range(query_num):
        if query_path.endswith('.jpg'):
            img=cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(query_inputs[i], cv2.IMREAD_GRAYSCALE)
        histogram = calculate_lbp(img)
        hists.append(histogram)
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
        
        save_flag, cumulative_histogram, histss=judge_memory(2,frame_count,cache)
        if save_flag==0:
            fg_mask = backSub.apply(image)
            # 将掩码应用于原始图像以获得彩色前景
            cumulative_histogram, histss=calculate(fg_mask,image)
            cache=update_memory(cache,2,frame_count,cumulative_histogram, histss)

        # fg_mask = backSub.apply(image)
        #     # 将掩码应用于原始图像以获得彩色前景
        # cumulative_histogram, histss=calculate(fg_mask,image)
        
        for i in range(query_num):
            similarity = compare_histograms(cumulative_histogram, hists[i])
            similarity_min = similarity
            for j in range(len(histss)):
                similarity_t = compare_histograms(histss[j], hists[i])
                if similarity_min > similarity_t:
                    similarity_min = similarity_t
            if similarity_min < 0.6:
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
    lbp_extraction()
