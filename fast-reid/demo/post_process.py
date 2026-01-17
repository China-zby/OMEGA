import os
import re
from numpy import nonzero
from pyparsing import null_debug_action
import torch
from collections import Counter
from torch.multiprocessing import Pool

# 定义处理单个文件的函数
def process_file(file_path):
    detections = []
    
    # 从文件名中提取帧的序号
    frame_number = int(file_path.split('_')[-1].split('.')[0])

    with open(file_path, 'r') as file:
        lines = file.readlines()
        car_num = truck_num = bus_num=1
        for line in lines:
            data = line.strip().split(' ')
            img_path=0
            track_feat = []
            similar = []
            if len(data) >= 6:
                category = int(data[0])
                bbox = list(map(float, data[1:5]))
                confidence = float(data[5])
                track_id = -1
            if len(data) >= 7:
                track_id = int(data[6])
                if(category==2):
                    img_path = file_path.replace("/labels/", "/crops/car/")
                    if(car_num==1):
                        img_path =os.path.dirname(img_path) + "/" +os.path.splitext(os.path.basename(file_path))[0] + ".jpg"
                    if(car_num!=1):
                        img_path =os.path.dirname(img_path) + "/" +os.path.splitext(os.path.basename(file_path))[0] +"-"+ str(car_num) + ".jpg"
                    car_num+=1
                if(category==5):
                    img_path = file_path.replace("/labels/", "/crops/bus/")
                    if(bus_num==1):
                        img_path =os.path.dirname(img_path) + "/" +os.path.splitext(os.path.basename(file_path))[0] + ".jpg"
                    if(bus_num!=1):
                        img_path =os.path.dirname(img_path) + "/" +os.path.splitext(os.path.basename(file_path))[0] +"-"+ str(bus_num) + ".jpg"
                    bus_num+=1
                if(category==7):
                    img_path = file_path.replace("/labels/", "/crops/truck/")
                    if(truck_num==1):
                        img_path =os.path.dirname(img_path) + "/" +os.path.splitext(os.path.basename(file_path))[0] + ".jpg"
                    if(truck_num!=1):
                        img_path =os.path.dirname(img_path) + "/" +os.path.splitext(os.path.basename(file_path))[0] +"-"+ str(truck_num) + ".jpg"
                    truck_num+=1
            detection = {
                "category": category,
                "bbox": bbox,
                "confidence": confidence,
                "track_id": track_id,
                "frame_number": frame_number,
                "file_name": os.path.basename(file_path),
                "camid":'_'.join(os.path.basename(file_path).split('_')[:-1]),
                "img_path":img_path,
                "min_frame":0,
                "max_frame":0,
                "feat":1,
                "feat_np":1,
                "track_feat":track_feat,
                "similar":similar,
                "exist":False
            }

            detections.append(detection)
    return detections

def extract_before_first_digit(s):
    match = re.search(r'^[^\d]*(?=\d)', s)
    if match:
        return match.group(0)
    else:
        return None

def process_detections(detections):
    # 统计每个 track_id 下 categories 的出现次数
    track_categories = {}
    track_frame_numbers = {}  # 用于记录每个 track_id 的 frame_number 范围

    for detection in detections:
        track_id = detection["track_id"]
        category = detection["category"]
        frame_number = detection["frame_number"]

        if track_id not in track_categories:
            track_categories[track_id] = Counter()
            
        if track_id not in track_frame_numbers:
            track_frame_numbers[track_id] = {"min_frame": frame_number, "max_frame": frame_number}
        else:
            track_frame_numbers[track_id]["min_frame"] = min(track_frame_numbers[track_id]["min_frame"], frame_number)
            track_frame_numbers[track_id]["max_frame"] = max(track_frame_numbers[track_id]["max_frame"], frame_number)

        track_categories[track_id][category] += 1

    # 更新 sorted_detections 中每个 track_id 的 categories
    for detection in detections:
        track_id = detection["track_id"]

        if track_id in track_categories:
            most_common_category = track_categories[track_id].most_common(1)[0][0]
            detection["category"] = most_common_category

    # 检查并更新 track_id
    for track_id, frame_info in track_frame_numbers.items():
        if frame_info["max_frame"] - frame_info["min_frame"] <= 15:
            # 更新为 -1
            for detection in detections:
                if detection["track_id"] == track_id:
                    detection["track_id"] = -1
        for detection in detections:
            if detection["track_id"] == track_id:
                detection["min_frame"]=frame_info["min_frame"]
                detection["max_frame"]=frame_info["max_frame"]

    return detections

def post_process_frame(txt_directory_input):
    txt_directory = txt_directory_input

    # 获取所有 txt 文件的完整路径列表
    txt_files = [os.path.join(txt_directory, filename) for filename in os.listdir(txt_directory) if filename.endswith(".txt")]

    # 使用多进程并行处理
    with Pool() as pool:
        all_detections = pool.map(process_file, txt_files)

    # 将多个进程的结果合并成一个列表
    all_detections = [detection for sublist in all_detections for detection in sublist]
    
    # 处理 detections，更新 categories
    all_detections = process_detections(all_detections)
    sorted_detections = sorted(all_detections, key=lambda x: x.get('frame_number', ''))

    return sorted_detections


def post_process_getdetection(txt_directory_in):
    txt_directory = txt_directory_in

    # 获取所有 txt 文件的完整路径列表
    txt_files = [os.path.join(txt_directory, filename) for filename in os.listdir(txt_directory) if filename.endswith(".txt")]

    # 使用多进程并行处理
    with Pool() as pool:
        all_detections = pool.map(process_file, txt_files)

    # 将多个进程的结果合并成一个列表
    all_detections = [detection for sublist in all_detections for detection in sublist]
    
    # 处理 detections，更新 categories
    all_detections = process_detections(all_detections)
    sorted_detections = sorted(all_detections, key=lambda x: x.get('frame_number', ''))  

    return sorted_detections

# if __name__ == '__main__':
#     # 指定包含 txt 文件的目录
#     txt_directory = "/mnt/data_hdd1/zby/track/runs/detect/predict/labels"

#     # 获取所有 txt 文件的完整路径列表
#     txt_files = [os.path.join(txt_directory, filename) for filename in os.listdir(txt_directory) if filename.endswith(".txt")]

#     # 使用多进程并行处理
#     with Pool() as pool:
#         all_detections = pool.map(process_file, txt_files)

#     # 将多个进程的结果合并成一个列表
#     all_detections = [detection for sublist in all_detections for detection in sublist]
    
#     # 处理 detections，更新 categories
#     track_frame_numbers = {}
#     all_detections = process_detections(all_detections)
#     sorted_detections = sorted(all_detections, key=lambda x: x.get('frame_number', ''))  

#     unique_track_ids = set(detection.get("track_id", None) for detection in sorted_detections)
#     num_unique_track_ids = len(unique_track_ids)

#     print(f"Number of unique track_ids: {num_unique_track_ids}")
#     print("Unique track_ids:", unique_track_ids)


