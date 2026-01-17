import glob
import os
import cv2
import pickle

query_path = '/home/lzp/zby/query_datasets/query_total_pet'
txt_path = '/home/lzp/zby/opencv/output/video_flags_11epochs_total_pet.txt'
frame_info_list = []  
flag_counts = []
positions_within_range = []
query_inputs = glob.glob(os.path.join(query_path, "*"))
for index in range(len(query_inputs)):
    query_path = query_inputs[index]
    
    if query_path.endswith('.jpg'):
        image = cv2.imread(query_path)
    else:
        image = cv2.imread(query_inputs[index])
    
    # 计算帧相关信息
    query_frame = int(os.path.basename(query_inputs[index]).split('.')[0].split('_')[-1].split('-')[0])
    frame_len = int(os.path.basename(query_inputs[index]).split('.')[0].split('_')[-3])
    min_frame = query_frame - frame_len // 2
    max_frame = query_frame + frame_len // 2
    # 存储帧相关信息
    frame_info = {
        'query_path': query_path,  # 保存路径
        'query_frame': query_frame,
        'frame_len': frame_len,
        'min_frame': min_frame,
        'max_frame': max_frame
    }
    frame_info_list.append(frame_info)

# 读取 video_flags.txt 文件并处理
with open(txt_path, 'r') as f:
    for index, line in enumerate(f):
        # 将字符串转换为列表
        flags = eval(line.strip())  # 使用 eval 将字符串转换为列表
        count_of_ones = flags.count(1)  # 计算 1 的数量
        flag_counts.append(count_of_ones)
        
        # 记录 1 的位置
        indices_of_ones = [i for i, value in enumerate(flags) if value == 1]
        
        # 获取当前帧的 min_frame 和 max_frame
        min_frame = frame_info_list[index]['min_frame']
        max_frame = frame_info_list[index]['max_frame']
        
        # 检查这些位置是否在 min_frame 和 max_frame 之间
        indices_in_range = [i for i in indices_of_ones if min_frame <= i <= max_frame]
        positions_within_range.append(indices_in_range)

# 输出每个 flag 中 1 的数量及其在范围内的位置
for index, (count, positions) in enumerate(zip(flag_counts, positions_within_range)):
    file_path = frame_info_list[index]['query_path']  # 获取对应的文件路径
    print(f"{file_path}: {count} 个 1，位置在范围内: {positions}")
    


import pickle

# 假设 frame_info_list 已经在您的代码中定义
# frame_info_list = [...]

# 首先确定总帧数
total_frames = 0
with open(txt_path, 'r') as f:
    for line in f:
        flags = eval(line.strip())
        if len(flags) > total_frames:
            total_frames = len(flags)

# 初始化一个 2000 x total_frames 的二维数组，初始值为 0
loaded_arrays = [[0] * total_frames for _ in range(4000)]

with open(txt_path, 'r') as f:
    for index, line in enumerate(f):
        flags = eval(line.strip())  # 将字符串转换为列表
        # 从 frame_info_list 获取帧信息
        min_frame = frame_info_list[index]['min_frame']
        max_frame = frame_info_list[index]['max_frame']
        file_path = frame_info_list[index]['query_path']
        
        # 从文件路径中提取倒数第二个数字
        filename = os.path.basename(file_path)
        parts = filename.rstrip('.jpg').split('_')
        second_last_number = int(parts[-2])  # 转换为整数
        
        # 确保索引在 [0, 1999] 范围内
        if 0 <= second_last_number < 4000:
            # 将 flags 中的值复制到 loaded_arrays 中对应的位置
            for i in range(total_frames):
                loaded_arrays[second_last_number][i] = flags[i]
        else:
            print(f"索引 {second_last_number} 超出范围，文件名为 {filename}")

# 将 loaded_arrays 保存到一个 pickle 文件中
with open('/home/lzp/zby/opencv/output/video_flags_11epochs_total_pet.pkl', 'wb') as file:
    pickle.dump(loaded_arrays, file)
