#!/bin/bash

# 视频路径数组
video_paths=(
    "/mnt/data_hdd1/zby/Seattle_50/1_Yesler_EW_11_12_7_50.mp4"
    "/mnt/data_hdd1/zby/Seattle_50/1_S_Lander_NS_11_12_7_50.mp4"
    "/mnt/data_hdd1/zby/Seattle_50/1_S_RoyalB_EW_11_12_7_50.mp4"
    "/mnt/data_hdd1/zby/Seattle_50/4_Madison_NS_11_12_7_50.mp4"
    "/mnt/data_hdd1/zby/Seattle_50/4_S_Jackson_NS_11_12_7_50.mp4"
    # 在这里添加其他视频路径
)

# 配置 YOLO 模型和参数
model="/home/lzp/zby/weight_v8_1/yolov8x.pt"
tracker="bytetrack.yaml"
conf=0.25
iou=0.6
classes="[2,5,7]"
device=0
save_txt=true
save_crop=true
save_conf=true

# 遍历每个视频路径
for video_path in "${video_paths[@]}"; do
    echo "Processing video: $video_path"
    
    # 构建命令
    command="time yolo track model=$model tracker=$tracker source=$video_path conf=$conf iou=$iou classes=$classes device=$device save_txt=$save_txt save_crop=$save_crop save_conf=$save_conf"
    
    # 执行命令
    eval $command
    
    # 执行完一个视频后停顿 1 秒
    sleep 1
done
