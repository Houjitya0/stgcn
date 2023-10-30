import numpy as np
import os
import time
import datetime
from utils.pre_data import to_center, t_skeleton_normarization_fixed_size, setup_data

def create_two_person_data(meta, try_count, input_file_name):
    
    print("create date_file")
    start = time.time()
        
    # 学習パラメータ
    NUM_CLASSES = meta["NUM_CLASSES"]

    # 短動画パラメータ
    interval_num = meta["interval_num"]
    min_move_num = meta["min_move_num"]
    out_frame_num = meta["out_frame_num"]
    
    shoulder_and_hip = meta["shoulder_and_hip"]
    left_node_index = meta["left_node_index"]
    right_node_index = meta["right_node_index"]
    
    if not (os.path.isdir(f"my_data/{input_file_name}/{try_count}")):
        keypoints = np.load("my_data/N_M_T_V_C_keiypoints.npy")
        labels = np.load("my_data/N_Time.npy")
        N, M, T, V, C = keypoints.shape
        VxM = len(left_node_index) + len(right_node_index)

        data = np.zeros((N * min_move_num, out_frame_num, VxM, C))

        for i in range(N):
            for j in range(M):
                # 人物をフレームの中心に持ってくる
                keypoints[i, j] = to_center(keypoints[i, j], shoulder_and_hip, 255, 255)
                # 時間
                keypoints[i, j] = t_skeleton_normarization_fixed_size(data=keypoints[i, j], fixed_size=50, is_center=meta["is_center"])

            # 左の人からはleft_node_indexにはいっているノードだけ取り出す
            # 例 left_node_index=[0, 1, 2, 15] 鼻,左目,右目,左足首
            # 左側の人と右側の人とを同じグラフにする
            left_person = np.transpose(keypoints[i, 0, :, left_node_index, :], (1, 0, 2))
            right_person = np.transpose(keypoints[i, 1, :, right_node_index, :], (1, 0, 2))
            
            tmp = np.concatenate((left_person, right_person), 1)

            data[i * min_move_num : (i + 1) * min_move_num] = setup_data(tmp, labels[i], out_frame_num, interval_num, min_move_num)

        data = data / meta["normarize_scale"]
        print(data.shape)

        # 正解ラベル作成
        label = np.arange(NUM_CLASSES)
        label = label.repeat(min_move_num)
        label = np.tile(label, (N // NUM_CLASSES))
        print(label.shape)

        # データを保存
        os.makedirs(name=f"my_data/{input_file_name}/{try_count}", exist_ok=True)

        np.save(f"my_data/{input_file_name}/{try_count}/data.npy", data)
        np.save(f"my_data/{input_file_name}/{try_count}/label.npy", label)

    else:
        print("data_file is existed")
        
        
    data_creation_time = time.time()
    print("data_creation_time : ", datetime.timedelta(seconds=data_creation_time - start))