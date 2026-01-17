from collections import defaultdict
import glob
import pickle
import cv2
import torch
import tqdm
import torch.nn.functional as F
import faiss
import numpy as np
import random
from datetime import datetime, timedelta

from zmq import NULL
from baseline import prepare_model


def query1(car1ID, car2ID, cameraID):
    flag = 0
    flag1, car1ID = car_exist_in_camera(car1ID, cameraID)
    flag2, car2ID = car_exist_in_camera(car2ID, cameraID)
    flag = flag1 & flag2
    return flag

def car_exist_in_camera(carID, cameraID, method):
    similarity_threshold = 0.002
    flag = False
    if method=='baseline':
        if(cameraID == '1_hour_Wyoming'):
            with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_Wyoming_1h_final.pkl', 'rb') as file:
                detections = pickle.load(file)
        if(cameraID == 'Gather_1hour'):
            with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_Gather_1h_final.pkl', 'rb') as file:
                detections = pickle.load(file)
        if(cameraID == 'Roadhouse_1hour'):
            with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_Roadhouse_1h_final.pkl', 'rb') as file:
                detections = pickle.load(file)
        demo = prepare_model()
        query_feats = []
        feat = demo.run_on_image(cv2.imread(carID))
        feat = F.normalize(feat)
        feat = feat.cpu().data.numpy()
        feat = torch.from_numpy(feat).numpy()
        feat = feat.flatten()
        query_feats.append(feat)
        # 获取所有的特征向量，并检查它们的长度
        feats = [detection["feat_np"].flatten() for detection in detections if detection["track_id"] > 0]
        gallery_names = [detection["track_id"] for detection in detections if detection["track_id"] > 0]
        gallery_paths = [detection["img_path"] for detection in detections if detection["track_id"] > 0]

        feats = np.array(feats)
        query_feats = np.array(query_feats)

        # 创建 Faiss 索引
        index = faiss.IndexFlatL2(feats.shape[1])  # 使用L2距离度量
        index.add(feats)
        # 查询最相似的 K 个向量
        k = 50
        distances, indices = index.search(query_feats, k)
        gallery_name_stats = {}
        for i in range(k):
            similarity = distances[0][i]
            gallery_name = gallery_names[indices[0][i]]
            if gallery_name not in gallery_name_stats:
                gallery_name_stats[gallery_name] = {"total_similarity": 0, "total_weight": 0, "total_num":0}
            # 使用相似度作为权重
            weight = similarity
            gallery_name_stats[gallery_name]["total_similarity"] += similarity * weight
            gallery_name_stats[gallery_name]["total_weight"] += weight
            gallery_name_stats[gallery_name]["total_num"] += 1
        output_q=0
        result = []
        for gallery_name, stats in gallery_name_stats.items():
            if stats["total_weight"] > 0 and stats["total_num"] > 3:
                weighted_avg_distance = stats["total_similarity"] / stats["total_weight"]
                #print(f"Weighted Average Distance for {gallery_name}: {weighted_avg_distance}")
                if(weighted_avg_distance<similarity_threshold):
                    flag = True
                    result.append(gallery_name)
        return flag, result
        
def query2(cameraID, carID, method):
    """carID是指定车辆的集合"""
    """这里默认是传入两辆车的数据"""
    query_IDs=[]
    times = defaultdict(list)
    for query_car in range(carID):
        exist, query_ID = car_exist_in_camera(query_car, cameraID)
        if(exist == False):
            print("the car is not exist in the current camera")
            break
        query_IDs.append(query_ID)
        for ID in query_ID:
            start_time, end_time = time_of_ID(ID, cameraID, method)
            times[ID].append((start_time,end_time))
    intersection_set = set(times[query_ID[0]])
    for time_range in times.values():
        intersection_set = intersection_set.intersection(set(time_range))

    # 将结果转换为列表并排序
    intersection_list = sorted(list(intersection_set))
    return intersection_list
    
def time_of_ID(carID, cameraID, method):
    if method == 'baseline':
        with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_1h_3.pkl', 'rb') as file:
            detections = pickle.load(file)
        for detection in detections:
            if(str(carID) == str(detection.get('track_id'))):
                if(str(cameraID)==str(detection.get('camid'))):
                    return detection.get('min_frame'), detection.get('max_frame')
            
def query3(carID, method):
    flag, car1ID = car_exist_in_camera(carID, 0)
    cameraIDs = []
    if method == 'baseline':
        with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_1h_3.pkl', 'rb') as file:
            detections = pickle.load(file)
        for ID in car1ID:
            for detection in detections:
                if(str(ID) == str(detection.get('track_id'))):
                    cameraIDs.append(detection.get('camid') )
    return cameraIDs

def query4(start_time, end_time, carIDs, method):
    camera_ids = ['1_hour_Wyoming', 'Clinic_1hour', 'Gather_1hour', 'Roadhouse_1hour']
    result = []
    for cameraID in camera_ids:
        for query_car in range(carIDs):
            flag, car1ID = car_exist_in_camera(query_car, cameraID)
            if method == 'baseline':
                with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_1h_3.pkl', 'rb') as file:
                    detections = pickle.load(file)
                for ID in car1ID:
                    for detection in detections:
                        if(str(ID) == str(detection.get('track_id'))):
                            if(detection.get('min_frame')<end_time or detection.get('miax_frame')>start_time):
                                result.append(cameraID)
    return result

def query5(carID, method):
    cameraIDs = {}
    flag, car1ID = car_exist_in_camera(carID, 0)
    if method == 'baseline':
        with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_1h_3.pkl', 'rb') as file:
            detections = pickle.load(file)
        for ID in car1ID:
            for detection in detections:
                if(str(ID) == str(detection.get('track_id'))):
                    cameraIDs.append((detection.get('camid'),detection.get('min_frame')))
    return cameraIDs

def random_sample_q1():
    camera_ids = ['1_hour_Wyoming', 'Gather_1hour', 'Roadhouse_1hour']
    random_camera_id = random.choice(camera_ids)
    query_path = '/mnt/data_hdd1/zby/jackson_town/datasets/query/'
    query_inputs = glob.glob(query_path + "*")  # 获取输入路径下所有的文件路径
    query_car1 = random.sample(query_inputs)
    query_car2 = random.sample(query_inputs)
    while(True):
        if(query_car1 != query_car2):
            break
        query_car2 = random.sample(query_inputs)
    print(query_car1)
    print(query_car2)
    print(random_camera_id)

    query1(query_car1, query_car2, random_camera_id)

def random_sample_q2():
    args=NULL
    camera_ids = ['1_hour_Wyoming', 'Gather_1hour', 'Roadhouse_1hour']
    random_camera_id = random.choice(camera_ids)
    query_path = '/mnt/data_hdd1/zby/jackson_town/datasets/query/'
    query_inputs = glob.glob(query_path + "*")  # 获取输入路径下所有的文件路径
    query_car1 = random.sample(query_inputs)
    query_car2 = random.sample(query_inputs)
    while(True):
        if(query_car1 != query_car2):
            break
        query_car2 = random.sample(query_inputs)
    carID = []
    carID.append(query_car1)
    carID.append(query_car2)
    query2(random_camera_id, carID, args)

def random_sample_q3():
    args=NULL
    query_path = '/mnt/data_hdd1/zby/jackson_town/datasets/query/'
    query_inputs = glob.glob(query_path + "*")  # 获取输入路径下所有的文件路径
    query_car1 = random.sample(query_inputs)
    query3(query_car1, args)

def random_sample_q4(query_num):
    args=NULL
    query_path = '/mnt/data_hdd1/zby/jackson_town/datasets/query/'
    query_inputs = glob.glob(query_path + "*")  # 获取输入路径下所有的文件路径
    query_car = random.sample(query_inputs, query_num)
    query4(0, 10000, query_car, args)

def random_sample_q5():
    args=NULL
    query_path = '/mnt/data_hdd1/zby/jackson_town/datasets/query/'
    query_inputs = glob.glob(query_path + "*")  # 获取输入路径下所有的文件路径
    query_car1 = random.sample(query_inputs)
    query5(query_car1, args)

def frame_to_time(first_time, frameID, fps):
    time = (frameID-1)/fps
    frame_datetime = first_time + timedelta(seconds=time)
    return frame_datetime


if __name__ == '__main__':
    for i in range(10):
        random_sample_q1()