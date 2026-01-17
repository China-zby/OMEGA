# def judge_memory(extractor_num,frame_count,cache):
#     if frame_count in cache:
#         color_1,color_2,energy_1,energy_2,lbp_1,lbp_2,color_lbp_1,color_lbp_2,color_energy_1,color_energy_2,glcm_lbp_1,glcm_lbp_2,net_1,net_2=cache[frame_count]
#     else:
#         cache[frame_count]=0,0,0,0,0,0,0,0,0,0,0,0,0,0
#     color_1,color_2,energy_1,energy_2,lbp_1,lbp_2,color_lbp_1,color_lbp_2,color_energy_1,color_energy_2,glcm_lbp_1,glcm_lbp_2,net_1,net_2=cache[frame_count]
#     if extractor_num==0:
#         if color_1==0:
#             return 0,0,0
#         return 1,color_1,color_2
#     if extractor_num==1:
#         if energy_1==0:
#             return 0,0,0
#         return 1,energy_1,energy_2
#     if extractor_num==2:
#         if lbp_1==0:
#             return 0,0,0
#         return 1,lbp_1,lbp_2
#     if extractor_num==3:
#         if color_lbp_1==0:
#             return 0,0,0
#         return 1,color_lbp_1,color_lbp_2
#     if extractor_num==4:
#         if color_energy_1==0:
#             return 0,0,0
#         return 1,color_energy_1,color_energy_2
#     if extractor_num==5:
#         if glcm_lbp_1==0:
#             return 0,0,0
#         return 1,glcm_lbp_1,glcm_lbp_2
#     if extractor_num==6:
#         if glcm_lbp_1==0:
#             return 0,0,0
#         return 1,net_1,net_2

# def update_memory(cache,extractor_num,frame_count,para_1,para_2):
#     color_1,color_2,energy_1,energy_2,lbp_1,lbp_2,color_lbp_1,color_lbp_2,color_energy_1,color_energy_2,glcm_lbp_1,glcm_lbp_2,net_1,net_2=cache[frame_count]
#     if extractor_num==0:
#         color_1=para_1
#         color_2=para_2

#     if extractor_num==1:
#         energy_1=para_1
#         energy_2=para_2

#     if extractor_num==2:
#         lbp_1=para_1
#         lbp_2=para_2

#     if extractor_num==3:
#         color_lbp_1=para_1
#         color_lbp_2=para_2

#     if extractor_num==4:
#         color_energy_1=para_1
#         color_energy_2=para_2

#     if extractor_num==5:
#         glcm_lbp_1=para_1
#         glcm_lbp_2=para_2

#     if extractor_num==6:
#         net_1=para_1
#         net_2=para_2
    
#     cache[frame_count]=color_1,color_2,energy_1,energy_2,lbp_1,lbp_2,color_lbp_1,color_lbp_2,color_energy_1,color_energy_2,glcm_lbp_1,glcm_lbp_2,net_1,net_2
#     return cache
import pickle


def judge_memory(extractor_num, frame_count, cache):
    if frame_count not in cache:
        # 初始化缓存
        cache[frame_count] = {
            'color': (0, 0),
            'energy': (0, 0),
            'lbp': (0, 0),
            'color_lbp': (0, 0),
            'color_energy': (0, 0),
            'glcm_lbp': (0, 0),
            'net': (0, 0)
        }
    
    # 获取缓存中的值
    feature_keys = ['color', 'energy', 'lbp', 'color_lbp', 'color_energy', 'glcm_lbp', 'net']
    feature_key = feature_keys[extractor_num]
    feature_1, feature_2 = cache[frame_count][feature_key]
    
    if isinstance(feature_1, (int, float)) or feature_1 == []:
        return 0, 0, 0
    return 1, feature_1, feature_2

def update_memory(cache, extractor_num, frame_count, para_1, para_2):
    cache_path = '/home/lzp/zby/opencv/cache/taipei_cache.pkl'
    # 获取缓存中的值
    feature_keys = ['color', 'energy', 'lbp', 'color_lbp', 'color_energy', 'glcm_lbp', 'net']
    feature_key = feature_keys[extractor_num]
    
    # 更新缓存
    cache[frame_count][feature_key] = (para_1, para_2)
    # with open(cache_path, 'wb') as f:
    #     pickle.dump(cache, f)
    return cache
