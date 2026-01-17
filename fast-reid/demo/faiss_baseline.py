from collections import defaultdict
import glob
import os
import pickle
import shutil
import cv2
import faiss
import numpy as np
import torch
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import distance

def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features

def faiss_baseline(detections, demo, query_path, query_num):
    query_feats = []
    query_names = []
    query_ids = []
    query_number = query_num
    query_lens =[]
    query_inputs = glob.glob(query_path + "/*")  # 获取输入路径下所有的文件路径
    for path in tqdm.tqdm(query_inputs):  # 逐张处理
        if(query_num<=0):
            break
        query_num-=1
        img = cv2.imread(path)
        feat = demo.run_on_image(img)
        feat = F.normalize(feat)
        feat = feat.cpu().data.numpy()
        feat = torch.from_numpy(feat).numpy()
        feat = feat.flatten()
        query_feats.append(feat)
        pid = os.path.basename(path).split('_')[-2]
        len_q = os.path.basename(path).split('_')[-3]
        query_lens.append(len_q)
        query_names.append(path)
        query_ids.append(pid)

    feats = []
    gallery_names = []
    gallery_paths = []

    for detection in detections:
        if detection["track_id"] > 0:
            feats.append(detection["feat_np"].flatten())
            gallery_names.append(detection["track_id"])
            gallery_paths.append(detection["img_path"])


    # # 获取所有的特征向量，并检查它们的长度
    # feats = [detection["feat_np"].flatten() for detection in detections if detection["track_id"] > 0]
    # gallery_names = [detection["track_id"] for detection in detections if detection["track_id"] > 0]
    # gallery_paths = [detection["img_path"] for detection in detections if detection["track_id"] > 0]

    feats = np.array(feats)
    query_feats = np.array(query_feats)

    # 创建 Faiss 索引
    index = faiss.IndexFlatL2(feats.shape[1])  # 使用L2距离度量
    index.add(feats)

    # 假设你有一个查询特征向量 query_feat
    # query_feats = np.array([your_query_feature_vector_1, your_query_feature_vector_2, ..., your_query_feature_vector_50])

    # 查询最相似的 K 个向量
    k = 50
    distances, indices = index.search(query_feats, k)

    # 统计满足相似度阈值的正确匹配结果
    threshold_range = np.linspace(0.001, 0.02, 50)
    f1_scores = []
    # # 循环遍历不同的相似度阈值
    for similarity_threshold in threshold_range:
    # 创建一个字典来存储每个query的结果匹配轨迹
        query_result={} 


        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0   
        total_precise = 0
        total_recall = 0
        # 打印结果
        for q in range(query_feats.shape[0]):
            # 创建一个字典来存储每个gallery_name的累计相似度和权重
            gallery_name_stats = {}
            query_name = query_names[q]  # 获取当前查询的名称
            query_id = query_ids[q]  # 获取当前查询的名称
            #print(f"Results for {query_name}:")
            for i in range(k):
                similarity = distances[q][i]
                gallery_name = gallery_names[indices[q][i]]
                gallery_path = gallery_paths[indices[q][i]]
                if gallery_name not in gallery_name_stats:
                    gallery_name_stats[gallery_name] = {"total_similarity": 0, "total_weight": 0, "total_num":0}
                # 使用相似度作为权重
                weight = similarity
                gallery_name_stats[gallery_name]["total_similarity"] += similarity * weight
                gallery_name_stats[gallery_name]["total_weight"] += weight
                gallery_name_stats[gallery_name]["total_num"] += 1
            #print("\n")
            if query_id not in query_result:
                query_result[query_id] = []
            # 计算每个gallery_name的加权平均距离
            output_q=0
            for gallery_name, stats in gallery_name_stats.items():
                if stats["total_weight"] > 0 and stats["total_num"] > 3:
                    weighted_avg_distance = stats["total_similarity"] / stats["total_weight"]
                    #print(f"Weighted Average Distance for {gallery_name}: {weighted_avg_distance}")
                    if(weighted_avg_distance<similarity_threshold):
                        query_result[query_id].append(gallery_name)
                        #print(f"Weighted Average Distance for {gallery_name}: {weighted_avg_distance}")
                        output_q+=1
                else:
                    continue

            if(output_q==0):
                print("no match track")
            true_positive = 0
            false_positive = 0
            precise_1 = 0
            recall_1 = 0
            for track_id in query_result[query_id]:
                if(str(query_id)==str(track_id)):
                    true_positive += 1
                    recall_1 = 1
                else:
                    false_positive += 1    
            if((false_positive + true_positive) != 0):
                precise_1 = true_positive/(false_positive + true_positive) 
            total_precise += precise_1
            total_recall += recall_1

        precision = total_precise/query_number
        recall = total_recall/query_number
        if(precision + recall!=0):
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0

        print(precision)
        print(recall)
        print(f1_score)
        print("\n")

        f1_scores.append(f1_score)
    # 保存数据到文件
    data_to_save = np.column_stack((threshold_range, f1_scores))
    savepath = 'threshold_f1_scores' + query_path.split('/')[-1] + '.csv'
    np.savetxt(savepath, data_to_save, delimiter=',')
    # 绘制阈值与f1 score的关系图
    plt.plot(threshold_range, f1_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Similarity Threshold')
    plt.grid(True)
    plt.show()

def make_label(detections):
    image_features = {}
    for detection in detections:
        if detection["track_id"]<=0:
            continue
        name = str(detection["camid"])+"_"+str(detection["track_id"])
        if name in image_features:
            continue
        image_features[name] = detection["track_feat"]

    threshold = 0.0003

    # 创建一个列表，其中每个元素都是一组图片的特征向量数组
    features_groups = defaultdict(list)
    for group in image_features.keys():
        # 假设每个特征向量都是相同维度的 numpy 数组
        if len(image_features[group]) != 0:
            group_features = np.array(image_features[group][0])
            group_features = group_features.reshape(1, -1)
            features_groups[group].append(group_features)

    # 创建 FAISS 索引
    d = features_groups[list(features_groups.keys())[0]][0].shape[1]  # 特征的维度
    index = faiss.IndexFlatL2(d)  # 使用 L2 距离

    # 创建一个列表来记录每个特征向量属于哪一组
    group_ids = []
    for group_name, features in features_groups.items():
        index.add(np.vstack(features))  # 将每组的特征向量添加到索引中
        group_ids.extend([group_name] * len(features))  # 对于每组的每个特征，记录其组ID

    group_ids = np.array(group_ids)

    # 检查哪些组是同一个物体
    n, _ = index.ntotal, index.d
    D, I = index.search(index.reconstruct_n(0, n), n)  # 搜索所有特征向量

    similar_group=defaultdict(list)
    for i in range(n):
        for j in range(1, n):  # 跳过自己，检查其它邻居
            if group_ids[i] != group_ids[I[i, j]] and D[i, j] <= threshold:
                print(f"图片组 {group_ids[i]} 和 图片组 {group_ids[I[i, j]]} 可能是同一个物体 {D[i, j]}")
                similar_group[group_ids[i]].append(group_ids[I[i, j]])
    for track_name in similar_group.keys():
        for detection in detections:
            id = track_name.split('_')[-1]
            camid = track_name.split('_')[:-1]
            camid = '_'.join(camid)
            if(str(detection["camid"]) == camid and str(detection["track_id"]) == id):
                detection["similar"]=similar_group[track_name]
                break
    return detections



    #定义相似度阈值的范围和数量
    # threshold_range = np.linspace(0.0002, 0.003, 40)
    # f1_scores = []

    # # 循环遍历不同的相似度阈值
    # for threshold in threshold_range:
    #     print(threshold)
    #     total_true_positives = 0
    #     total_false_positives = 0
    #     total_false_negatives = 0   
    #     # 打印结果
    #     for q in range(query_feats.shape[0]):
    #         query_name = query_names[q]  # 获取当前查询的名称
    #         query_id = query_ids[q]  # 获取当前查询的名称
    #         true_positives=0
    #         false_positives=0
    #         false_negatives=0
    #         #print(f"Results for {query_name}:")
    #         for i in range(k):

    #             # 判断是否满足相似度阈值
    #             if distances[q][i] < threshold:
    #                 if(str(query_id)==str(gallery_names[indices[q][i]])):
    #                     true_positives += 1
    #                 else:
    #                     false_positives += 1            
    #             else: 
    #                 if(int(query_lens[q])>k):
    #                     false_negatives = k - true_positives
    #                 else:
    #                     false_negatives = int(query_lens[q]) - true_positives
    #                 break
    #             #print("Similarity:", distances[q][i])
    #             #print("Corresponding Detection:", gallery_names[indices[q][i]], gallery_path[indices[q][i]])
    #         #print(true_positives)
    #         #print(false_positives)
    #         #print(false_negatives)
    #         #print("\n")

    #         total_true_positives += true_positives
    #         total_false_positives += false_positives
    #         total_false_negatives += false_negatives

    #     precision = total_true_positives / (total_true_positives + total_false_positives)
    #     recall = total_true_positives / (total_true_positives + total_false_negatives)

    #     f1_score = 2 * (precision * recall) / (precision + recall)
    #     print(precision)
    #     print(recall)
    #     print(f1_score)
    #     print("\n")

    #     f1_scores.append(f1_score)
    # # 保存数据到文件
    # data_to_save = np.column_stack((threshold_range, f1_scores))
    # np.savetxt('threshold_f1_scores.csv', data_to_save, delimiter=',')
    # # 绘制阈值与f1 score的关系图
    # plt.plot(threshold_range, f1_scores, marker='o', linestyle='-', color='b')
    # plt.xlabel('Similarity Threshold')
    # plt.ylabel('F1 Score')
    # plt.title('F1 Score vs Similarity Threshold')
    # plt.grid(True)
    # plt.show()

        # 定义相似度阈值的范围和数量
    #threshold_range = np.linspace(int(query_lens[q]), k, k-query_lens[q])
    # f1_scores = []
    # # 循环遍历不同的相似度阈值

    # total_true_positives = 0
    # total_false_positives = 0
    # total_false_negatives = 0   
    # # 打印结果
    # for q in range(query_feats.shape[0]):
    #     threshold = int(query_lens[q]) //2
    #     print(threshold)
    #     query_name = query_names[q]  # 获取当前查询的名称
    #     query_id = query_ids[q]  # 获取当前查询的名称
    #     true_positives=0
    #     false_positives=0
    #     false_negatives=0
    #     print(f"Results for {query_name}:")
    #     for i in range(k):

    #         # 判断是否满足相似度阈值
    #         if i < threshold:
    #             if(str(query_id)==str(gallery_names[indices[q][i]])):
    #                 true_positives += 1
    #             else:
    #                 false_positives += 1            
    #         else: 
    #             false_negatives = int(query_lens[q]) - true_positives
    #             break
    #         print("Similarity:", distances[q][i])
    #         print("Corresponding Detection:", gallery_names[indices[q][i]], gallery_path[indices[q][i]])
    #     print(true_positives)
    #     print(false_positives)
    #     print(false_negatives)
    #     print("\n")

    #     total_true_positives += true_positives
    #     total_false_positives += false_positives
    #     total_false_negatives += false_negatives

    # precision = total_true_positives / (total_true_positives + total_false_positives)
    # recall = total_true_positives / (total_true_positives + total_false_negatives)

    # f1_score = 2 * (precision * recall) / (precision + recall)
    # print(precision)
    # print(recall)
    # print(f1_score)
    # print("\n")

    # f1_scores.append(f1_score)

def faiss_baseline_test_filter(detections, demo, query_path, query_num,video_flag,index,q_index):
    query_feats = []
    query_names = []
    query_ids = []
    query_number = query_num
    query_lens =[]
    query_inputs = glob.glob(query_path + "/*")  # 获取输入路径下所有的文件路径
    for path in tqdm.tqdm(query_inputs):  # 逐张处理
        if int(os.path.basename(path).split('_')[-2])!=index:
            continue
        img = cv2.imread(path)
        feat = demo.run_on_image(img)
        feat = F.normalize(feat)
        feat = feat.cpu().data.numpy()
        feat = torch.from_numpy(feat).numpy()
        feat = feat.flatten()
        query_feats.append(feat)
        pid = os.path.basename(path).split('_')[-2]
        len_q = os.path.basename(path).split('_')[-3]
        query_lens.append(len_q)
        query_names.append(path)
        query_ids.append(pid)
        break
    

    feats = []
    gallery_names = []
    gallery_paths = []
    count=0
    for detection in detections:
        if detection["track_id"] > 0:
            if(video_flag[int(detection["frame_number"])]==1):
                count=count+1
                feats.append(detection["feat_np"].flatten())
                gallery_names.append(detection["track_id"])
                gallery_paths.append(detection["img_path"])
 

    # # 获取所有的特征向量，并检查它们的长度
    # feats = [detection["feat_np"].flatten() for detection in detections if detection["track_id"] > 0]
    # gallery_names = [detection["track_id"] for detection in detections if detection["track_id"] > 0]
    # gallery_paths = [detection["img_path"] for detection in detections if detection["track_id"] > 0]

    feats = np.array(feats)
    query_feats = np.array(query_feats)

    # 创建 Faiss 索引
    index = faiss.IndexFlatL2(feats.shape[1])  # 使用L2距离度量
    index.add(feats)

    # 假设你有一个查询特征向量 query_feat
    # query_feats = np.array([your_query_feature_vector_1, your_query_feature_vector_2, ..., your_query_feature_vector_50])

    # 查询最相似的 K 个向量
    k = 50
    distances, indices = index.search(query_feats, k)

    # 统计满足相似度阈值的正确匹配结果
    threshold_range = np.linspace(0.1, 0.1, 1)
    f1_scores = []
    # # 循环遍历不同的相似度阈值
    for similarity_threshold in threshold_range:
    # 创建一个字典来存储每个query的结果匹配轨迹
        query_result={} 


        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0   
        total_precise = 0
        total_recall = 0
        # 打印结果
        for q in range(query_feats.shape[0]):
            # 创建一个字典来存储每个gallery_name的累计相似度和权重
            gallery_name_stats = {}
            query_name = query_names[q]  # 获取当前查询的名称
            query_id = query_ids[q]  # 获取当前查询的名称
            query_folder = "/mnt/data_hdd1/zby/jackson_town/result/query_" + str(query_id)
            if os.path.exists(query_folder):
                shutil.rmtree(query_folder)
            os.makedirs(query_folder, exist_ok=True)
            shutil.copy(query_name, os.path.join(query_folder, f"query_for_{query_id}.jpg"))

            for i in range(k):
                similarity = distances[q][i]
                if similarity<=0.000001:
                    continue
                gallery_name = gallery_names[indices[q][i]]
                gallery_path = gallery_paths[indices[q][i]]
                feat = feats[indices[q][i]]
                new_filename = f"{i}_{similarity}_{gallery_name}.jpg"
                #print(f"Weighted Average Distance for {gallery_path}: {similarity}")
                shutil.copy(gallery_path, os.path.join(query_folder, new_filename))
                if gallery_name not in gallery_name_stats:
                    gallery_name_stats[gallery_name] = {"total_similarity": 0, "total_weight": 0, "total_num":0, "mean_feat":feat}
                # 使用相似度作为权重
                weight = i
                gallery_name_stats[gallery_name]["mean_feat"] += feat
                gallery_name_stats[gallery_name]["total_similarity"] += similarity
                gallery_name_stats[gallery_name]["total_weight"] += weight
                gallery_name_stats[gallery_name]["total_num"] += 1
            #print("\n")
            if query_id not in query_result:
                query_result[query_id] = []
            # 计算每个gallery_name的加权平均距离
            output_q=0
            for gallery_name, stats in gallery_name_stats.items():
                if stats["total_weight"] > 0 and stats["total_num"] > 3:
                    weighted_avg_distance = stats["total_similarity"] / stats["total_num"] * stats["total_weight"] / stats["total_num"]
                    mean_f = stats["mean_feat"] / stats["total_num"]
                    avg_distance = distance.cosine(mean_f, query_feats[q])
                    #print(f"Weighted Average Distance for {gallery_name}: {weighted_avg_distance}")
                    if(weighted_avg_distance<similarity_threshold):
                        query_result[query_id].append(gallery_name)
                        print(f"Weighted Average Distance for {gallery_name}: {weighted_avg_distance}")
                        output_q+=1
                else:
                    continue


            true_positive = 0
            false_positive = 0
            precise_1 = 0
            recall_1 = 0
            for track_id in query_result[query_id]:
                if(str(query_id)==str(track_id)):
                    true_positive += 1
                    recall_1 = 1
                else:
                    false_positive += 1    
            if((false_positive + true_positive) != 0):
                precise_1 = true_positive/(false_positive + true_positive) 
            total_precise += precise_1
            total_recall += recall_1

        precision = total_precise/query_number
        recall = total_recall/query_number
        if(precision + recall!=0):
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0



        f1_scores.append(f1_score)
    # 保存数据到文件
    data_to_save = np.column_stack((threshold_range, f1_scores))
    savepath = 'threshold_f1_scores' + query_path.split('/')[-1] + '.csv'
    np.savetxt(savepath, data_to_save, delimiter=',')
    # 绘制阈值与f1 score的关系图
    plt.plot(threshold_range, f1_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Similarity Threshold')
    plt.grid(True)
    plt.show()
    print('/n')
    return f1_score


def faiss_baseline_test(detections, demo, query_path, query_num):
    query_feats = []
    query_names = []
    query_ids = []
    query_number = query_num
    query_lens =[]
    query_inputs = glob.glob(query_path + "/*")  # 获取输入路径下所有的文件路径
    for path in tqdm.tqdm(query_inputs):  # 逐张处理
        if(query_num<=0):
            break
        query_num-=1
        img = cv2.imread(path)
        feat = demo.run_on_image(img)
        feat = F.normalize(feat)
        feat = feat.cpu().data.numpy()
        feat = torch.from_numpy(feat).numpy()
        feat = feat.flatten()
        query_feats.append(feat)
        pid = os.path.basename(path).split('_')[-2]
        len_q = os.path.basename(path).split('_')[-3]
        query_lens.append(len_q)
        query_names.append(path)
        query_ids.append(pid)

    feats = []
    gallery_names = []
    gallery_paths = []

    for detection in detections:
        if detection["track_id"] > 0:
            feats.append(detection["feat_np"].flatten())
            gallery_names.append(detection["track_id"])
            gallery_paths.append(detection["img_path"])


    # # 获取所有的特征向量，并检查它们的长度
    # feats = [detection["feat_np"].flatten() for detection in detections if detection["track_id"] > 0]
    # gallery_names = [detection["track_id"] for detection in detections if detection["track_id"] > 0]
    # gallery_paths = [detection["img_path"] for detection in detections if detection["track_id"] > 0]

    feats = np.array(feats)
    query_feats = np.array(query_feats)

    # 创建 Faiss 索引
    index = faiss.IndexFlatL2(feats.shape[1])  # 使用L2距离度量
    index.add(feats)

    # 假设你有一个查询特征向量 query_feat
    # query_feats = np.array([your_query_feature_vector_1, your_query_feature_vector_2, ..., your_query_feature_vector_50])

    # 查询最相似的 K 个向量
    k = 50
    distances, indices = index.search(query_feats, k)

    # 统计满足相似度阈值的正确匹配结果
    threshold_range = np.linspace(0.005, 0.1, 100)
    f1_scores = []
    # # 循环遍历不同的相似度阈值
    for similarity_threshold in threshold_range:
    # 创建一个字典来存储每个query的结果匹配轨迹
        query_result={} 


        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0   
        total_precise = 0
        total_recall = 0
        # 打印结果
        for q in range(query_feats.shape[0]):
            # 创建一个字典来存储每个gallery_name的累计相似度和权重
            gallery_name_stats = {}
            query_name = query_names[q]  # 获取当前查询的名称
            query_id = query_ids[q]  # 获取当前查询的名称
            query_folder = "/mnt/data_hdd1/zby/jackson_town/result/query_" + str(query_id)
            if os.path.exists(query_folder):
                shutil.rmtree(query_folder)
            os.makedirs(query_folder, exist_ok=True)
            shutil.copy(query_name, os.path.join(query_folder, f"query_for_{query_id}.jpg"))
            print(f"Results for {query_name}:")
            for i in range(k):
                similarity = distances[q][i]
                if similarity<=0.000001:
                    continue
                gallery_name = gallery_names[indices[q][i]]
                gallery_path = gallery_paths[indices[q][i]]
                feat = feats[indices[q][i]]
                new_filename = f"{i}_{similarity}_{gallery_name}.jpg"
                #print(f"Weighted Average Distance for {gallery_path}: {similarity}")
                shutil.copy(gallery_path, os.path.join(query_folder, new_filename))
                if gallery_name not in gallery_name_stats:
                    gallery_name_stats[gallery_name] = {"total_similarity": 0, "total_weight": 0, "total_num":0, "mean_feat":feat}
                # 使用相似度作为权重
                weight = i
                gallery_name_stats[gallery_name]["mean_feat"] += feat
                gallery_name_stats[gallery_name]["total_similarity"] += similarity
                gallery_name_stats[gallery_name]["total_weight"] += weight
                gallery_name_stats[gallery_name]["total_num"] += 1
            #print("\n")
            if query_id not in query_result:
                query_result[query_id] = []
            # 计算每个gallery_name的加权平均距离
            output_q=0
            for gallery_name, stats in gallery_name_stats.items():
                if stats["total_weight"] > 0 and stats["total_num"] > 3:
                    weighted_avg_distance = stats["total_similarity"] / stats["total_num"] * stats["total_weight"] / stats["total_num"]
                    mean_f = stats["mean_feat"] / stats["total_num"]
                    avg_distance = distance.cosine(mean_f, query_feats[q])
                    #print(f"Weighted Average Distance for {gallery_name}: {weighted_avg_distance}")
                    if(weighted_avg_distance<similarity_threshold):
                        query_result[query_id].append(gallery_name)
                        print(f"Weighted Average Distance for {gallery_name}: {weighted_avg_distance}")
                        output_q+=1
                else:
                    continue

            if(output_q==0):
                print("no match track")
            true_positive = 0
            false_positive = 0
            precise_1 = 0
            recall_1 = 0
            for track_id in query_result[query_id]:
                if(str(query_id)==str(track_id)):
                    true_positive += 1
                    recall_1 = 1
                else:
                    false_positive += 1    
            if((false_positive + true_positive) != 0):
                precise_1 = true_positive/(false_positive + true_positive) 
            total_precise += precise_1
            total_recall += recall_1

        precision = total_precise/query_number
        recall = total_recall/query_number
        if(precision + recall!=0):
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0

        print(precision)
        print(recall)
        print(f1_score)
        print("\n")

        f1_scores.append(f1_score)
    # 保存数据到文件
    data_to_save = np.column_stack((threshold_range, f1_scores))
    savepath = 'threshold_f1_scores' + query_path.split('/')[-1] + '.csv'
    np.savetxt(savepath, data_to_save, delimiter=',')
    # 绘制阈值与f1 score的关系图
    plt.plot(threshold_range, f1_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Similarity Threshold')
    plt.grid(True)
    plt.show()