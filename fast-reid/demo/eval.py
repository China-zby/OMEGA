import os
import pickle
import re


def eval_rank(query_id, top5_result):
    top1 = True
    for item in top5_result:
        if top1 == True:
            if str(query_id)==str(item['track_id']):
                rank1=1
                rank5=1
            else: 
                rank1=0
                rank5=0
            top1=False
        if str(query_id)==str(item['track_id']):
            rank5=1
        else: 
            rank5=0
    return rank1,rank5

# # 计算 F1 Score 的函数
# def calculate_f1_score(true_labels, indices, k):
#     total_true_positives = 0
#     total_false_positives = 0
#     total_false_negatives = 0

#     for q in range(true_labels.shape[0]):
#         true_positives = len(set(true_labels[q]).intersection(set(indices[q])))
#         false_negatives = k[q] - true_positives
#         false_positives = len(true_labels[q]) - true_positives

#         total_true_positives += true_positives
#         total_false_positives += false_positives
#         total_false_negatives += false_negatives

#     precision = total_true_positives / (total_true_positives + total_false_positives)
#     recall = total_true_positives / (total_true_positives + total_false_negatives)

#     f1_score = 2 * (precision * recall) / (precision + recall)

#     return f1_score

#     # 假设每个查询有一个真实标签，存储在 true_labels 中
#     true_labels = np.array([true_labels_for_query_1, true_labels_for_query_2, ..., true_labels_for_query_50])

#     # 计算 F1 Score
#     f1_score = calculate_f1_score(true_labels, indices, k)

#     print("F1 Score:", f1_score)

def eval_query1(car1ID, car2ID, cameraID, result, query_num):
    precise= []
    recall = []
    with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_Wyoming_1h_final.pkl', 'rb') as file:
        detections = pickle.load(file)
    for i in range(query_num):
        if(cameraID[i] == '1_hour_Wyoming'):
            precise[i] = result[i]
            recall[i] = result[i]
        else:
            car1_cam = re.search(r'^[^\d]*(?=\d)', os.path.basename(car1ID[i])).group(0)
            car1_id = car1ID[i].split('_')[4]
            car2_cam = re.search(r'^[^\d]*(?=\d)', os.path.basename(car2ID[i])).group(0)
            car2_id = car2ID[i].split('_')[4]
            flag1 = False
            flag2 = False 
            find1 = 0
            find2 = 0
            for detection in detections:
                if(str(detection["track_id"]) == str(car1_id)):
                    find1=1
                    if(car1_cam in detection["similar"].keys()):
                        flag1 = True
                if(str(detection["track_id"]) == str(car2_id)):
                    find2=1
                    if(car2_cam in detection["similar"].keys()):
                        flag2 = True
                if(find1*find2 == 1):
                    break
            if(result[i] == 1 and flag1 == True and flag2 == True):
                precise[i] = 1
                recall[i] = 1
            if(result[i] == 1 and (flag1&flag2) == False):
                precise[i] = 0
                recall[i] = 0
            if(result[i] == 0 and flag1 == True and flag2 == True):
                precise[i] = 0
                recall[i] = 0
            if(result[i] == 0 and (flag1&flag2) == False):
                precise[i] = 1
                recall[i] = 1
    sum_p=0
    sum_r=0
    for i in range(query_num):
        sum_p+=precise[i]
        sum_r=recall[i]
    return sum_p/query_num, sum_r/query_num

def eval_query2(car1ID, car2ID, cameraID, result, query_num):
    precise= []
    recall = []
    with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_Wyoming_1h_final.pkl', 'rb') as file:
        detections = pickle.load(file)
    for i in range(query_num):
        car1_cam = re.search(r'^[^\d]*(?=\d)', os.path.basename(car1ID[i])).group(0)
        car1_id = car1ID[i].split('_')[4]
        car2_cam = re.search(r'^[^\d]*(?=\d)', os.path.basename(car2ID[i])).group(0)
        car2_id = car2ID[i].split('_')[4]
        flag1 = False
        flag2 = False 
        find1 = 0
        find2 = 0
        for detection in detections:
            if(str(detection["track_id"]) == str(car1_id)):
                find1=1
                detection["min_frame"]

def eval_query3(carID, result, query_num):
    precise= []
    recall = []
    with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_Wyoming_1h_final.pkl', 'rb') as file:
        detections = pickle.load(file)
    for i in range(query_num):
        car1_cam = re.search(r'^[^\d]*(?=\d)', os.path.basename(carID[i])).group(0)
        car1_id = carID[i].split('_')[4]
        for detection in detections:
            if(str(detection["track_id"]) == str(car1_id)):
                if("Wyoming_syn_" in result[i]):
                    precise[i]=1
                    recall[i]=1
                else:
                    precise[i]=0
                    recall[i]=0
    sum_p=0
    sum_r=0
    for i in range(query_num):
        sum_p+=precise[i]
        sum_r=recall[i]
    return sum_p/query_num, sum_r/query_num

def eval_query4():
    1
def eval_query5(carID, result, query_num):
    precise= []
    recall = []
    with open('/mnt/data_hdd1/zby/track/fast-reid/demo/detections_Wyoming_1h_final.pkl', 'rb') as file:
        detections = pickle.load(file)
    for i in range(query_num):
        car1_cam = re.search(r'^[^\d]*(?=\d)', os.path.basename(carID[i])).group(0)
        car1_id = carID[i].split('_')[4]
        for detection in detections:
            if(str(detection["track_id"]) == str(car1_id)):
                if(detection["similar"].size()==0):
                    if((detection["camid"],detection["min_frame"])in result[i]):
                        precise[i]=1/result[i].len()
                        recall[i]=1
