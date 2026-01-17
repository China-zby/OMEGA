import argparse
import glob
import os
import sys
import torch
import time
import pickle

import numpy as np
import torch.nn.functional as F
import cv2
import tqdm
from torch.backends import cudnn
from faiss_baseline import faiss_baseline, faiss_baseline_test,faiss_baseline_test_filter
from eval import eval_rank

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger

from predictor import FeatureExtractionDemo
from pre_process_datasets import process_datasets

# import some modules added in project like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--test_baseline', type=lambda x: (str(x).lower() == 'true'), default=True, help="Test baseline or not")
    return parser


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features

def prepare_model():
    args = get_parser().parse_args()
    
    # ---
    args.config_file = "/mnt/data_hdd1/zby/track/fast-reid/configs/VERIWild/bagtricks_R50-ibn.yml" 
    # ---

    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    return demo

def baseline_one_img(demo, img_path, detections, save_flag, save_path):

    start_time = time.time()
    orig_img_path = img_path
    img = cv2.imread(img_path)
    feat = demo.run_on_image(img)
    feat = postprocess(feat)
    feat = torch.from_numpy(feat)
    pid = os.path.basename(img_path).split('_')[1]
    camid = os.path.basename(img_path).split('_')[0]
    frameid = int(os.path.basename(img_path).split('_')[-1].split('-')[0].split('.')[0])
    similarity_matrix=[]

    for detection in tqdm.tqdm(detections):
        track_id = detection.get('track_id')
        if(track_id<0):
            continue
        img_path = detection.get('img_path')
        frame_number = detection.get('frame_number')
        if(detection["exist"] == False):
            img = cv2.imread(img_path)
            feat_d = demo.run_on_image(img)
            feat_d = postprocess(feat_d)
            feat_d = torch.from_numpy(feat_d)
            detection["feat"] = feat_d
            detection["feat_np"] = feat_d.numpy()
            detection["exist"] = True
            # add new
            # if(detection["frame_number"] >= (detection["min_frame"]+(detection["max_frame"]-detection["min_frame"])//4) and detection["frame_number"] <= (detection["max_frame"]-(detection["max_frame"]-detection["min_frame"])//4)):
            #     detection["track_feat"] = detection["track_feat"].append(feat_d.numpy())
        else:
            feat_d = detection["feat"]
        if(frame_number>frameid+5 or frame_number<frameid-5):
            similarity = compute_similarity_1to1(feat,feat_d)
            similarity_matrix_item = {
                "similarity_degree":similarity[0][0],
                "track_id":track_id,
                "img_path":img_path
            }
            similarity_matrix.append(similarity_matrix_item)
    similarity_matrix = sorted(similarity_matrix, key=sort_by_similarity_degree)
    end_time = time.time()
    print(f"query time: {end_time - start_time} 秒")
    print(orig_img_path)
    if save_flag == False:
        with open(save_path, 'wb') as file:
            pickle.dump(detections, file)
            print("complete")
            save_flag = True
    top_five_items = similarity_matrix[:5]
    for item in top_five_items:
        print(item)

    return top_five_items, pid, save_flag

def compute_similarity_1to1(feats1, feats2):
    feats1 = F.normalize(feats1, p=2, dim=1)
    feats2 = F.normalize(feats2, p=2, dim=1)
    distmat = 1 - torch.mm(feats1, feats2.t())
    return distmat.numpy() 

def sort_by_similarity_degree(item):
    return item["similarity_degree"]

def baseline_muti_img(save_path, demo, detections, query_path, query_num):
    query_inputs = glob.glob(query_path + "/*")  
    avg_rank1 = 0
    avg_rank5 = 0
    query_num_s = query_num
    save_flag =False
    for path in tqdm.tqdm(query_inputs): 
        if(query_num<=0):
            break
        query_num-=1
        one_result = []
        one_result, query_id, save_flag = baseline_one_img(demo, path, detections, save_flag, save_path)
        rank1,rank5=eval_rank(query_id, one_result)
        avg_rank1 += rank1
        avg_rank5 += rank5
    return avg_rank1/query_num_s,avg_rank5/query_num_s, detections

def test():

    # path = input("input folder path:")
    # print(path)
    args = get_parser().parse_args()
    # detections, query_path = process_datasets(path)
    query_path = '/mnt/data_hdd4/zby/query_video/query_img/query_total_pet'
    label_path = '/mnt/data_hdd4/zby/runs/detect/predict-3/labels'
    detections_path = '/mnt/data_hdd4/zby/query_video/cache/detections_' + 'total_pet' + '.pkl'

    test_baseline = args.test_baseline
    demo = prepare_model()
    if os.path.exists(detections_path):
        print("detections exist")
        with open(detections_path, 'rb') as file:
            detections = pickle.load(file)

    else:
        detections, query_path = process_datasets(label_path)
        save_path = '/mnt/data_hdd4/zby/query_video/cache/detections_' + detections[0]["camid"] + '.pkl'
        avg_rank1, avg_rank5, detections=baseline_muti_img(save_path, demo, detections, query_path, 1) 

        with open(save_path, 'wb') as file:
            pickle.dump(detections, file) 

    print(detections[0].get("track_feat"))
    print(detections[0].get("track_id"))
        
    s=time.time()
    if test_baseline == False:
        with open('/mnt/data_hdd1/zby/track/fast-reid/demo/test/video_flags_11epochs_total_pet.pkl', 'rb') as file:
            loaded_arrays = pickle.load(file)
        j=0
        sum_f1=0
        for i in range(3500):
            if np.sum(loaded_arrays[i]) > 2:
                print(i)
                if i == (3176 or 1594 or 2490):
                    continue
                f1 = faiss_baseline_test_filter(detections, demo, query_path,1,loaded_arrays[i],i,j)
                sum_f1 = sum_f1 + f1
                j=j+1
        print(time.time()-s)
        print(sum_f1/j)
    else:
        faiss_baseline_test(detections, demo, query_path, 12)
        print(time.time()-s)

if __name__ == '__main__':

    test()