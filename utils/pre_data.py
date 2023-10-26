import numpy as np
import random


import numpy as np
import math

def translate_data(data, trans_cood):

    data = data - trans_cood
    return data

# 線分を求める
def calc_seg(s1, s2, h1, h2):
    return (s1+s2) / 2, ((h1+h2) / 2)

# 角度を求める
def calculate_angle(m1, m2):
    # ベクトルABの成分を計算
    vector_ab = (m2[0] - m1[0],  m2[1] - m1[1])
    
    # ベクトルABの大きさを計算
    magnitude_ab = math.sqrt(vector_ab[0]**2 + vector_ab[1]**2)
    
    # 内積を計算
    dot_product = vector_ab[0] * 1 + vector_ab[1] * 0  # 1, 0はx軸のユニットベクトル
    
    # cosθを計算
    if magnitude_ab != 0:
        cos_theta = dot_product / magnitude_ab
    else:
        cos_theta = 0
    
    # ラジアンから度数に変換
    angle_in_radians = math.acos(cos_theta)
    angle_in_degrees = math.degrees(angle_in_radians)
    
    return angle_in_degrees


# 座標の角度を変換する
def rotate_point(p, angle_in_degrees):
    # 角度をラジアンに変換
    angle_in_radians = math.radians(angle_in_degrees)
    
    # 座標を回転させる
    x_rotated = p[0] * math.cos(angle_in_radians) - p[1] * math.sin(angle_in_radians)
    y_rotated = p[0] * math.sin(angle_in_radians) + p[1] * math.cos(angle_in_radians)
    
    return np.array([x_rotated, y_rotated])


# 距離を求める
def calculate_distance(p1, p2):
    distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return distance
    

# 座標を定数倍する
def scale_data(data, height, ratio, distance):
    if (distance == 0):
        return data
    else:
        scale_factor = (height * ratio) / (distance)
    
    data = data * scale_factor
    return data

def scale_data_fixed(data, fixed_size, distance):
    if (distance == 0):
        distance = distance + 0.001
    data = data * (fixed_size / distance)       
    return data




def skeleton_normarization(data, ratio, left_sholder, right_sholder, left_hip, right_hip, width=255, height=255):
    """
    骨格データを正規化する
    
    Parameters
    ----------
    data : ndarray (V, C)
    ratio : float
    left_sholder : int
    right_sholder : int
    left_hip : int
    right_hip : int
    
    Returns
    -------
    normarized_data : ndarray(V, C)
    """
    
    V, C = data.shape
    
    # キーポイントの中心をもと得る(両肩と両腰の平均)
    center = np.mean(data[[left_sholder, right_sholder, left_hip, right_hip],:], axis=0)
    
    # 原点へ移動 
    data = translate_data(data, center)
    
    # 中点を求める
    m1, m2 = calc_seg(data[left_sholder], data[right_sholder], data[left_hip], data[right_hip])
        
    # 角度を求める
    angle_in_degree = calculate_angle(m1, m2)
    
    # ノードごとに角度を変換する
    for i in range(len(data)):
        data[i] = rotate_point(data[i], 90 - angle_in_degree)
        
    # 距離を求める
    distance = calculate_distance(m1, m2)    
    
    # 距離を一定にする
    data = scale_data(data, height, ratio, distance)

    # もとの位置に戻る
    data = translate_data(data, -center)
    
    return data


def skeleton_normarization_fixed(data, fixed_size, left_sholder, right_sholder, left_hip, right_hip, is_center, width=255, height=255):
    """
    骨格データを正規化する
    
    Parameters
    ----------
    data : ndarray (V, C)
    ratio : float
    left_sholder : int
    right_sholder : int
    left_hip : int
    right_hip : int
    
    Returns
    -------
    normarized_data : ndarray(V, C)
    """
    
    V, C = data.shape
    
    # キーポイントの中心をもと得る(両肩と両腰の平均)
    center = np.mean(data[[left_sholder, right_sholder, left_hip, right_hip],:], axis=0)
    
    # 原点へ移動 
    data = translate_data(data, center)
    
    # 中点を求める
    m1, m2 = calc_seg(data[left_sholder], data[right_sholder], data[left_hip], data[right_hip])
        
    # 角度を求める
    angle_in_degree = calculate_angle(m1, m2)
    
    # ノードごとに角度を変換する
    for i in range(len(data)):
        data[i] = rotate_point(data[i], 90 - angle_in_degree)
        
    # 距離を求める
    distance = calculate_distance(m1, m2)    
    
    # 距離を一定にする
    data = scale_data_fixed(data, fixed_size, distance)

    # もとの位置に戻る
    if (is_center):
        return data
    else:
        data = translate_data(data, -center)
        return data


def t_skeleton_normarization_fixed_size(data, fixed_size, is_center=True, left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12, hasMidpoint=False, width=255, height=255):
    T, V, C = data.shape
    midpoints = np.empty((T, 2, C))
    normarized_data = np.empty((T, V, C))

    for i in range(len(data)):
        normarized_data[i] = skeleton_normarization_fixed(data[i], fixed_size, left_shoulder, right_shoulder, left_hip, right_hip, is_center, width, height)
    
    if (hasMidpoint):
        mid_normarized_data = np.empty((T, V+2, C))
        for i in range(len(data)):
            s1 = normarized_data[i, left_shoulder]
            s2 = normarized_data[i, right_shoulder]
            h1 = normarized_data[i, left_hip]
            h2 = normarized_data[i, right_hip]            
            
            m1, m2 = calc_seg(s1, s2, h1, h2)
            print(calculate_angle(m1, m2))
            mid_normarized_data[i, :V] = normarized_data[i]
            mid_normarized_data[i, V] = np.array(m1)
            mid_normarized_data[i, V+1] = np.array(m2)
            

        return mid_normarized_data
    else:
        return normarized_data
    


def t_skeleton_normarization(data, ratio, left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12, hasMidpoint=False, width=255, height=255):
    
    """
    骨格データを正規化する
    
    Parameters
    ----------
    data : ndarray (T, V, C)
        時系列の骨格データ
    ratio : float 
        両肩の中点と両肩の中点を結んだ線の画像の高さに対しての比率
    left_shoulder : int
        左肩のキーポイントのインデックス
    right_shoulder : int
        右肩
    left_hip : int
        左腰
    right_hip : int
        右腰
    hasMidpoint : bool, default False
        中点を含んだキーポイントを返すかどうか
    width : int, default 255
        画像の横幅
    height : int, default 255
        画像の縦幅 : int, default 255
        
    Returns
    -------
    normarized_data : ndarray(T, V, C)
    """
    
    T, V, C = data.shape
    midpoints = np.empty((T, 2, C))
    normarized_data = np.empty((T, V, C))

    for i in range(len(data)):
        normarized_data[i] = skeleton_normarization(data[i], ratio, left_shoulder, right_shoulder, left_hip, right_hip, width, height)
    
    if (hasMidpoint):
        mid_normarized_data = np.empty((T, V+2, C))
        for i in range(len(data)):
            s1 = normarized_data[i, left_shoulder]
            s2 = normarized_data[i, right_shoulder]
            h1 = normarized_data[i, left_hip]
            h2 = normarized_data[i, right_hip]            
            
            m1, m2 = calc_seg(s1, s2, h1, h2)
            print(calculate_angle(m1, m2))
            mid_normarized_data[i, :V] = normarized_data[i]
            mid_normarized_data[i, V] = np.array(m1)
            mid_normarized_data[i, V+1] = np.array(m2)
            

        return mid_normarized_data
    else:
        return normarized_data


def to_center(data, specified_keypoints_index, width, height):
    
    
    center = np.mean(data[:,specified_keypoints_index,:], axis=1)

    # 移動先の位置を指定
    move_to = np.array([width // 2, height // 2])

    # 中心からの相対位置を計算して、全ての点を移動
    data_moved = data - center[:, np.newaxis, :] + move_to

    return data_moved





# 一人のデータに[T, V, C]に対して, [min_mov_num, T, V, C]のデータ作成
# in_frame_numは12秒の360フレーム
def setup_data(one_person_data, in_frame_num, out_frame_num, interval_num, min_mov_num):
    T, V, C = one_person_data.shape
    mim_mov_data = np.zeros((min_mov_num, out_frame_num, V, C))
    
    # 動画の開始時刻の最低限の開始位置を計算
    rand_lim = in_frame_num - (interval_num * out_frame_num)

    # 0から最低限の開始位置までの値から短動画生成数だけ重複なしでランダムに取り出す
    head_list = random.sample(range(rand_lim), min_mov_num)
    
    for i, head in enumerate(head_list):
        time_list = np.arange(head, head + (out_frame_num * interval_num), interval_num)
    
        mim_mov_data[i] = one_person_data[time_list]

    return mim_mov_data


# [T, V, C]に対して, [min_mov_num, T, V, C]のデータ作成
def create_short_video_and_adjust_framesize(video_keypoint, in_frame_num ,out_frame_num, interval_num, min_mov_num):
    T, V, C = video_keypoint.shape
    
    mim_mov_data = np.zeros((min_mov_num, out_frame_num, V, 2))
    
    # 動画の開始時刻の最低限の開始位置を計算
    rand_lim = in_frame_num - (interval_num * out_frame_num)

    # 0から最低限の開始位置までの値から短動画生成数だけ重複なしでランダムに取り出す
    head_list = random.sample(range(rand_lim), min_mov_num)
    
    for i, head in enumerate(head_list):
        time_list = np.arange(head, head + (out_frame_num * interval_num), interval_num)
    
        mim_mov_data[i] = video_keypoint[time_list]

    return mim_mov_data

