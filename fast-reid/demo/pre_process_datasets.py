import os
import random
import re
import shutil
from PIL import Image
from post_process import post_process_getdetection


def process_datasets(file_path):

    detections=post_process_getdetection(file_path)
    track_id_to_img_paths = {}
    for detection in detections:
        track_id = detection.get('track_id')
        img_path = detection.get('img_path')
        if track_id is not None and img_path is not None and track_id>0:
            if track_id in track_id_to_img_paths:
                track_id_to_img_paths[track_id].append(img_path)
            else:
                track_id_to_img_paths[track_id] = [img_path]
            
    # 随机选择50个 track_id
    random_track_ids = random.sample(list(track_id_to_img_paths.keys()), 20)

    # 创建存储查询图像的文件夹
    query_folder = "/mnt/data_hdd4/zby/query_video/query_img/query_" + detections[0]["camid"]
    if os.path.exists(query_folder):
       shutil.rmtree(query_folder)
    os.makedirs(query_folder, exist_ok=True)
    # if not os.path.exists(query_folder):
    #     os.makedirs(query_folder)

    # 设定尺寸约束
    MIN_WIDTH = 70
    MIN_HEIGHT = 70
    # 遍历每个选定的 track_id
    for track_id in random_track_ids:
        # 找到该 track_id 对应的所有帧的信息
        img_paths = track_id_to_img_paths[track_id]

        # 找到中间帧的信息    
        middle_img_path = img_paths[len(img_paths) // 3]
        # 构造图像的新文件名
        match = re.match(r'(.*)_', os.path.basename(middle_img_path))
        if match:
            cam_name = match.group(1)
        else:
            print("未找到匹配的部分")
        frame_number = os.path.basename(middle_img_path).split('_')[-1].split('.')[0]
        new_filename = f"{cam_name}_{len(img_paths)}_{track_id}_{frame_number}.jpg"
        with Image.open(middle_img_path) as img:
                width, height = img.size
                
                # 判断图像尺寸是否符合约束
                if width >= MIN_WIDTH and height >= MIN_HEIGHT:
                    # 复制图像到查询文件夹并重命名
                    shutil.copy(middle_img_path, os.path.join(query_folder, new_filename))
                else:
                    continue

    return detections, query_folder

if __name__ == '__main__':
    path='/mnt/data_hdd4/zby/runs/detect/predict-3/labels'
    process_datasets(path) 