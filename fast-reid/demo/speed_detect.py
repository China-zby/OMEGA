from ast import Num
from post_process import extract_before_first_digit, post_process_getdetection
import math

def calculate_speed_between_detections(detection1, detection2, scale, frame_interval):
    # 提取位置信息
    x1, y1, w1, h1 = detection1["bbox"]
    x2, y2, w2, h2 = detection2["bbox"]

    # 计算两帧之间的时间差
    time_diff = detection2["frame_number"] - detection1["frame_number"]

    # 计算位置的变化量（像素）
    dx = x2 - x1
    dy = y2 - y1

    # 计算欧氏距离（像素）
    distance_pixel = math.sqrt(dx**2 + dy**2)

    # 将像素转换为实际距离（米）
    distance_meter = distance_pixel * scale

    # 计算速度（米/秒）
    speed = distance_meter / (time_diff * frame_interval) if time_diff != 0 else 0

    return speed

def calculate_average_speed(file_path, scale, fps):
    """计算每个track id对应的平均速度"""
    speeds = []
    detections=post_process_getdetection(file_path)

    # 字典用于存储每个 track id 对应的速度
    track_id_speeds = {}

    # 遍历每两个相邻的检测框
    for i in range(len(detections) - 1):
        detection1 = detections[i]
        detection2 = detections[i + 1]

        # 只计算相同track id的速度
        if detection1["track_id"] == detection2["track_id"]:
            speed = calculate_speed_between_detections(detection1, detection2, scale, fps)

            # 将速度添加到对应的 track id 中
            track_id = detection1["track_id"]
            if track_id not in track_id_speeds:
                track_id_speeds[track_id] = []

            track_id_speeds[track_id].append(speed)

    # 计算每个 track id 的平均速度
    average_speeds = {}
    for track_id, speeds in track_id_speeds.items():
        average_speed = sum(speeds) / len(speeds) if len(speeds) > 0 else 0
        average_speeds[track_id] = average_speed

    return average_speeds

def time_interval(distance, file_path, scale, fps):
    average_speeds = {}
    average_speeds = calculate_average_speed(file_path, scale, fps)
    for average_speed in average_speeds:
        sum += average_speed
        num += 1
    avg_speed = sum/num
    time=distance/avg_speed
    return time