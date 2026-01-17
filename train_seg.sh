#!/bin/bash

# 使用4/5切分的视频路径
video_path='/home/lzp/zby/video_segments/1_Yesler_EW_segment_4_5.mp4'

# 记录开始时间
start_time=$(date +%s)
echo "========================================="
echo "Script started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Video path: $video_path"
echo "========================================="
echo ""

python mix_reinforce_dpn_new.py \
    --input_epoch 9 \
    --query_path '/home/lzp/zby/query_datasets/query_1_Yesler_EW_11_12_7_50' \
    --video_path "$video_path" \
    --cache_path /home/lzp/zby/opencv/cache/Yesler_EW_11_12_7_50_4_5_cache_shell.pkl \
    --query_num 10

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "========================================="
echo "Script completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Duration: $duration seconds ($(date -u -d @${duration} +%H:%M:%S))"
echo "========================================="
