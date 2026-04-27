import os
import pickle
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt

# 直接导入你现有的数据集类，保证过滤逻辑和索引顺序 100% 对齐
from sample4geo.dataset.uavvisloc import UAVVisLocDatasetTrain

def haversine(lat1, lon1, lat2, lon2):
    """
    计算两点经纬度之间的实际地理距离（米）
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000 # 地球平均半径
    return c * r

def generate_gps_dict(dataset: UAVVisLocDatasetTrain, save_path: str, max_dist: float = 200):
    # dataset.all_samples_info 中存储的是: 
    # (drone_path, scene_id, x_pixel, y_pixel, lat, lon, label, phi)
    samples = dataset.all_samples_info
    n = len(samples)
    gps_dict = {}

    print(f"提取了 {n} 个有效训练样本，开始计算 {n}x{n} 的距离矩阵...")

    for i in tqdm(range(n)):
        lat1, lon1 = samples[i][4], samples[i][5] # 索引 4 和 5 对应 lat 和 lon
        neighbors = []

        for j in range(n):
            if i == j: 
                continue
            
            lat2, lon2 = samples[j][4], samples[j][5]
            dist = haversine(lat1, lon1, lat2, lon2)

            # 只记录设定距离内的样本（作为难负样本挖掘的候选池）
            if 50 <= dist <= max_dist:
                neighbors.append((j, dist))

        # 按照距离从小到大排序，方便后续优先抽取更近的难负样本
        neighbors.sort(key=lambda x: x[1])
        gps_dict[i] = neighbors

    total_neighbors = sum(len(v) for v in gps_dict.values())
    print(f"计算完成！平均每个训练图有 {total_neighbors/n:.2f} 个 {max_dist} 米内的邻居。")

    # 保存为 pkl，建议和作者提供的放一起
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(gps_dict, f)
    print(f"gps_dict 成功保存至: {save_path}")

if __name__ == '__main__':
    # ==========================================
    # 这里的配置必须与 train_uavvisloc.py 严格一致
    # ==========================================
    data_folder = '../Datasets/UAV_VisLoc_dataset'
    train_scene_ids = ['01', '02', '04', '05', '07', '08', '10', '11'] 
    
    print("正在初始化 UAVVisLocDatasetTrain (这可能需要一点时间来解析 TIF 文件并过滤非法样本)...")
    
    # 实例化 Dataset 类，拿到所有合法样本的数据。
    # 这里不需要传 transforms，因为我们仅仅是为了获取列表，不进行 __getitem__ 读取
    train_dataset = UAVVisLocDatasetTrain(
        data_folder=data_folder,
        scene_ids=train_scene_ids,
        transforms_query=None,
        transforms_reference=None
    )

    # 你可以把 pkl 保存在 MFRGN-pretained 预训练文件夹中，保持和源码结构一致
    save_path = 'pretrained/MFRGN-pretained/distance_dict/gps_dict_uavvisloc.pkl'
    
    # 判定为“难负样本”的最大距离范围
    max_hard_negative_dist = 500 
    
    generate_gps_dict(train_dataset, save_path, max_dist=max_hard_negative_dist)