# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
from itertools import combinations
import math
import os
import sys
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re

import torch.nn.functional as F
import cv2
import numpy as np
import tqdm
from torch.backends import cudnn

from post_process import extract_before_first_digit, post_process_getdetection

sys.path.append('.')

from fastreid.utils.visualizer import Visualizer
from fastreid.evaluation.rank import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from predictor import FeatureExtractionDemo

logger = logging.getLogger('fastreid.visualize_result')

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
    return parser


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features

def eval(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """
    评估使用Market1501指标的性能
    关键：对于每个查询身份，其来自相同相机视图的图库图像将被丢弃。
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('注意：图库样本数量相当少，获取 {}'.format(num_g))

    # 按距离矩阵排序，获取索引
    indices = np.argsort(distmat, axis=1)
    # 为每个查询计算cmc曲线
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # 有效查询数量

    for q_idx in range(num_q):
        # 获取查询的身份和相机ID
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # 移除与查询相同身份和相机ID的图库样本
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # 计算cmc曲线
        matches = (g_pids[order] == q_pid).astype(np.int32)
        raw_cmc = matches[keep]  # 二进制向量，值为1的位置是正确的匹配
        if not np.any(raw_cmc):
            # 当查询身份不出现在图库中时，此条件为真
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # 计算平均准确率
        # 参考：https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, '错误：所有查询身份均未出现在图库中'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP

def compute_similarity(feats1, feats2, num_images_to_use=3):
    """计算两组特征之间的相似度矩阵"""
 # 取每组特征中间几张图片
    feats1 = feats1[len(feats1)//2 - num_images_to_use//2 : len(feats1)//2 + num_images_to_use//2].mean(dim=0, keepdim=True)
    feats2 = feats2[len(feats2)//2 - num_images_to_use//2 : len(feats2)//2 + num_images_to_use//2].mean(dim=0, keepdim=True)
    feats1 = F.normalize(feats1, p=2, dim=1)
    feats2 = F.normalize(feats2, p=2, dim=1)
    distmat = 1 - torch.mm(feats1, feats2.t())
    return distmat.numpy() 


if __name__ == '__main__':
    args = get_parser().parse_args()
    
    # 调试使用，使用的时候删除下面代码
    # ---
    args.config_file = "/mnt/data_hdd1/zby/track/fast-reid/configs/VehicleID/bagtricks_R50-ibn.yml"  # config路径
    args.input = "/mnt/data_hdd1/zby/track/runs/detect/predict/crops/truck/"  # 图像路径
    # ---
    num_query=1
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)


    txt_directory = args.input.replace("/crops/truck/", "/labels/")
    detections=post_process_getdetection(txt_directory)

    PathManager.mkdirs(args.output)
    if args.input:
        if PathManager.isdir(args.input[0]):
            # args.input = glob.glob(os.path.expanduser(args.input[0])) # 原来的代码有问题
            args.input = glob.glob(os.path.expanduser(args.input))  # 获取输入路径下所有的文件路径
            assert args.input, "The input path(s) was not found"
            feats = []
            pids = []
            camids = []
        feats_set = []
        track_id_to_img_paths = {}
        for detection in detections:
            track_id = detection.get('track_id')
            img_path = detection.get('img_path')
            if track_id is not None and img_path is not None and track_id>0:
                if track_id in track_id_to_img_paths:
                    track_id_to_img_paths[track_id].append(img_path)
                else:
                    track_id_to_img_paths[track_id] = [img_path]
        for track_id, img_paths in tqdm.tqdm(track_id_to_img_paths.items()):
            feats = []
            pids = []
            camids = []
            for path in img_paths:
                img = cv2.imread(path)
                feat = demo.run_on_image(img)
                feat = postprocess(feat)
                feat = torch.from_numpy(feat)
                feats.append(feat)
                filename = os.path.basename(path)
                pids.append(filename)
                camids.append(1)
                camid=extract_before_first_digit(filename)
                print(os.path.basename(path))
            feats = torch.cat(feats, dim=0)
            feats_set_item={"track_id":track_id,
                       "feats":feats,
                       "camid":camid}
            feats_set.append(feats_set_item)

    # 创建一个字典用于存储所有 track_id 对应的特征
    track_id_to_feats = {(item["track_id"], item["camid"]): item["feats"] for item in feats_set}

    # 指定文件路径
    output_file_path = "/mnt/data_hdd1/zby/track/fast-reid/demo/output.txt"

    # 确保目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'a') as output_file:
        # 遍历所有组合，计算相似度矩阵
        for ((track_id1, camid1),feats1), ((track_id2, camid2),feats2) in combinations(track_id_to_feats.items(), 2):
        # 计算相似度矩阵
            similarity = compute_similarity(feats1, feats2)
            if(abs(similarity[0][0])<0.0011):
                matching_detection = next((detection for detection in detections if detection["track_id"] == track_id1 and detection["camid"] == camid1), None)
                min_frame = matching_detection["min_frame"]
                max_frame = matching_detection["max_frame"]
                matching_detection = next((detection for detection in detections if detection["track_id"] == track_id2 and detection["camid"] == camid2), None)
                min_frame2 = matching_detection["min_frame"]
                max_frame2 = matching_detection["max_frame"]
                if(camid1==camid2):
                    output_line = f"Track ID {track_id1} :frame {min_frame} to {max_frame} and Track ID {track_id2}:frame {min_frame2} to {max_frame2} is the same track in the same video\n"
                    output_file.write(output_line)
                if(camid1!=camid2):
                    output_line = f"Track ID {track_id1} :frame {min_frame} to {max_frame} and Track ID {track_id2}:frame {min_frame2} to {max_frame2} is the same track in different video\n"
                    output_file.write(output_line)
            print(f"Similarity between Track ID {track_id1} and Track ID {track_id2}: {similarity[0][0]}")

    for (track_id1, feats1) in track_id_to_feats.items():
    # 计算相似度矩阵``
        similarity = compute_similarity(feats1, feats1)
        print(f"Similarity between Track ID {track_id1} and Track ID {track_id1}: {similarity[0][0]}")


# top5_pids 中的每一行包含了查询图像与图库中前5个相似图像的身份 ID

# 计算各种评价指标 cmc[0]就是top1精度，应该是93%左右，这里精度会有波动
    """ logger.info("Computing APs for all query images ...")
    cmc, all_ap, all_inp = eval(distmat, q_pids, g_pids, q_camids, g_camids, 50)
    logger.info("Finish computing APs for all query images!")

    print(cmc[0]) """

