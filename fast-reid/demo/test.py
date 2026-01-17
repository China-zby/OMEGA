import os
import pickle
import random
import re
import shutil

from faiss_baseline import make_label

def test(path):
    with open(path, 'rb') as file:
        detections = pickle.load(file)

    track_id_to_feat = {}
    for detection in detections:
        track_id = detection.get('track_id')
        if(track_id<=0):
            continue
        if(detection["max_frame"]-detection["min_frame"]>2500):
            detection["track_id"] = -1
            continue
        if(detection["frame_number"] >= (detection["min_frame"]+(detection["max_frame"]-detection["min_frame"])//4) and detection["frame_number"] <= (detection["max_frame"]-(detection["max_frame"]-detection["min_frame"])//4)): 
            if track_id in track_id_to_feat:
                track_id_to_feat[track_id].append(detection["feat_np"][0])
            else:
                track_id_to_feat[track_id] = [detection["feat_np"][0]]
            
    for detection in detections:
        track_id = detection.get('track_id')
        if(track_id<=0):
            continue
        if track_id in track_id_to_feat:
            detection["track_feat"] = track_id_to_feat[track_id]
    return detections


if __name__ == '__main__':
    with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_Wyoming_1h_final.pkl', 'rb') as file:
        detections = pickle.load(file)
    query_folder = "/mnt/data_hdd1/zby/jackson_town/datasets/query_Wyoming"
    if os.path.exists(query_folder):
       shutil.rmtree(query_folder)
    os.makedirs(query_folder, exist_ok=True)
    # if not os.path.exists(query_folder):
    #     os.makedirs(query_folder)
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
    random_track_ids = random.sample(list(track_id_to_img_paths.keys()), 50)

    # 遍历每个选定的 track_id
    for track_id in random_track_ids:
        # 找到该 track_id 对应的所有帧的信息
        img_paths = track_id_to_img_paths[track_id]

        # 找到中间帧的信息
        middle_img_path = img_paths[len(img_paths) // 2]

        # 构造图像的新文件名
        match = re.match(r'(.*)_', os.path.basename(middle_img_path))
        if match:
            cam_name = match.group(1)
        else:
            print("未找到匹配的部分")
        frame_number = os.path.basename(middle_img_path).split('_')[-1].split('.')[0]
        new_filename = f"{cam_name}_{track_id}_{len(img_paths)}_{frame_number}.jpg"

        # 复制图像到查询文件夹并重命名
        shutil.copy(middle_img_path, os.path.join(query_folder, new_filename)) 
