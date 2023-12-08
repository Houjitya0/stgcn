import numpy as np
import os
import time
import datetime
from utils.pre_data import to_center, t_skeleton_normarization_fixed_size, setup_data, t_skeleton_perspectiveTransform, PreNormalize3D
from utils.preprocess import pre_normalization, transform_normarize

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
        
        if (meta["in_channels"] == 2):
            keypoints = np.load("my_data/N_M_T_V_C_keiypoints.npy")
            labels = np.load("my_data/N_Time.npy")


            
            N, M, T, V, C = keypoints.shape
            if meta["is_turned"] == True:
                keypoints[N//2:, :, :, :, 0] = keypoints[N//2:, :, :, :, 0] * -1.0
                

            

            VxM = len(left_node_index) + len(right_node_index)

            data = np.zeros((N * min_move_num, out_frame_num, VxM, C))

            for i in range(N):
                for j in range(M):
                    
                    if meta["has_bn"] == False:
                        # 人物をフレームの中心に持ってくる
                        keypoints[i, j] = to_center(keypoints[i, j], shoulder_and_hip, 255, 255)
                        # 時間(T, V, C)を入力にして正規化
                        keypoints[i, j] = t_skeleton_normarization_fixed_size(data=keypoints[i, j], fixed_size=50, is_center=meta["is_center"])
                        
                    if meta["has_perspective"] == True:
                        keypoints[i, j] = t_skeleton_perspectiveTransform(data=keypoints[i, j],  shoulder_and_hip=shoulder_and_hip)
                        

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


##################################### 3d ##############################################################################################
        elif (meta["in_channels"] in [3, 9]):
            
            if (meta["pose_data_type"] == "phalp"):
                keypoints = np.load("my_data/phalp_n_m_t_v_c.npy")
            elif (meta["pose_data_type"] == "phalp_rotation"):
                keypoints = np.load("my_data/phalp_rotation_n_m_t_v_c.npy")
            else:
                keypoints = np.load("my_data/3d_keiypoints.npy")
                
            labels = np.load("my_data/N_Time.npy")       
            
            N, M, T, V, C = keypoints.shape
            VxM = len(left_node_index) + len(right_node_index)
            data = np.zeros((N * min_move_num, out_frame_num, VxM, C))
            
            
            # pysklの軸を合わせる正規化
            if (meta["pyskl_3d_normarize"] == True):
                keypoints = pre_normalization(keypoints)
                
            if (meta["transform_normarize"] == True):
                keypoints = transform_normarize(keypoints, meta)
            
            tmp = keypoints.reshape(N, T, V*M, C)   
            for i in range(N):                
                
                # 左の人からはleft_node_indexにはいっているノードだけ取り出す
                # 例 left_node_index=[0, 1, 2, 15] 鼻,左目,右目,左足首
                # 左側の人と右側の人とを同じグラフにする
                left_person = np.transpose(keypoints[i, 0, :, left_node_index, :], (1, 0, 2))
                right_person = np.transpose(keypoints[i, 1, :, right_node_index, :], (1, 0, 2))
                
                tmp = np.concatenate((left_person, right_person), 1)
                data[i * min_move_num : (i + 1) * min_move_num] = setup_data(tmp, labels[i], out_frame_num, interval_num, min_move_num)

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
    
    
    
def create_one_person_data(meta, try_count, input_file_name):
    print("create date_file")
    start = time.time()
        
    # 学習パラメータ
    NUM_CLASSES = meta["NUM_CLASSES"]

    # 短動画パラメータ
    interval_num = meta["interval_num"]
    min_move_num = meta["min_move_num"]
    out_frame_num = meta["out_frame_num"]

    node_index = meta["node_index"]
    shoulder_and_hip = meta["shoulder_and_hip"]
    node_num = meta["node_num"]
    
    if not (os.path.isdir(f"my_data/{input_file_name}/{try_count}")):
        keypoints = np.load("my_data/N_M_T_V_C_keiypoints.npy")
        labels = np.load("my_data/N_Time.npy")
        N, M, T, V, C = keypoints.shape
        VxM = node_num

        data = np.zeros((N * min_move_num * 2, out_frame_num, V, C))

        keypoints = keypoints.reshpae(N*2, T, V, C)
        
        for i in range(N):
            # 人物をフレームの中心に持ってくる
            keypoints[i] = to_center(keypoints[i], shoulder_and_hip, 255, 255)
            # 時間
            keypoints[i] = t_skeleton_normarization_fixed_size(data=keypoints[i], fixed_size=50, is_center=meta["is_center"])

            data[i] = np.transpose(keypoints[i, 0, :, node_index, :], (1, 0, 2))
            
            # data[i * min_move_num : (i + 1) * min_move_num] = setup_data(tmp, labels[i], out_frame_num, interval_num, min_move_num)

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