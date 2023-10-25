import cv2
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
import datetime
import argparse
from utils.pre_data import setup_data, to_center, t_skeleton_normarization_fixed_size, t_skeleton_normarization
from pair_train import pair_train
from vote import pair_vote

##git用

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('config_file_name', type=str, help='config_file_name')
args = parser.parse_args()
config_file_name = args.config_file_name

with open(f"configs/{config_file_name}.json", "r") as json_file:
    meta = json.load(json_file)


# 実験識別ネーム
name = meta["name"]
node_option_name = meta["node_option_name"]
print("実験名", name)
print("ノードオプション", node_option_name if node_option_name != "" else "なし")

num_people = meta["num_people"]
node_index = meta["node_index"]

# 学習パラメータ
try_num = meta["try_num"]
BATCH_SIZE = meta["BATCH_SIZE"]
NUM_CLASSES = meta["NUM_CLASSES"]
NUM_EPOCH = meta["NUM_EPOCH"]
NUM_PAIR = meta["NUM_PAIR"]

# 短動画パラメータ
interval_num = meta["interval_num"]
min_move_num = meta["min_move_num"]
out_frame_num = meta["out_frame_num"]
print('\ninterval_num', interval_num)
print('min_move_num', min_move_num)
print('out_frame_num', out_frame_num)

if (num_people == 2):
    left_node_index = node_index[0]
    right_node_index = node_index[1]

# *****************変数定義*********************************************************************
input_file_name = f'{num_people}p_{node_option_name}_interval_{interval_num}_min_mov_num_{min_move_num}_out_frame_num_{out_frame_num}'
output_file_name = f'{num_people}p_{name}_{node_option_name}_interval_{interval_num}_min_mov_num_{min_move_num}_out_frame_num_{out_frame_num}'
shoulder_and_hip = [5, 6, 11, 12]



# ******************試行開始**********************************************************************
for try_count in range(1, try_num+1):
    
    os.makedirs(name=f'my_data/{input_file_name}', exist_ok=True)
    
# ************************データ作成****************************************************************
    print('create date_file')
    start = time.time()
    
    if not(os.path.isdir(f'my_data/{input_file_name}/{try_count}')):
        keypoints = np.load("my_data/N_M_T_V_C_keiypoints.npy")
        labels = np.load("my_data/N_Time.npy")
        N, M, T, V, C = keypoints.shape
        V = 17
        # それぞれの人にたいして正規化
        # tmp = np.zeros((N, T, M*V, C))
        data = np.zeros((N*min_move_num, out_frame_num, V*M, C))
        
        for i in range(N):
            for j in range(M):
                # 人物をフレームの中心に持ってくる
                keypoints[i, j] = to_center(keypoints[i, j], shoulder_and_hip, 255, 255) 
                # 時間
                keypoints[i, j] = t_skeleton_normarization(keypoints[i, j], ratio=0.2)
                
            # 左の人からはleft_node_indexにはいっているノードだけ取り出す
            # 例 left_node_index=[0, 1, 2, 15] 鼻,左目,右目,左足首
            # 左側の人と右側の人とを同じグラフにする

            left_person = np.transpose(keypoints[i, 0, :, left_node_index, :], (1, 0, 2))
            right_person = np.transpose(keypoints[i, 1, :, left_node_index, :], (1, 0, 2))
            tmp = np.concatenate((left_person, right_person), 1)

            data[i*min_move_num:(i+1)*min_move_num] = setup_data(tmp,labels[i], out_frame_num, interval_num, min_move_num)        
                
        print(data.shape)
        
        # 正解ラベル作成
        label = np.arange(NUM_CLASSES)
        label = label.repeat(min_move_num)
        label = np.tile(label, (N//NUM_CLASSES))
        print(label.shape)

        # データを保存
        os.makedirs(name=f'my_data/{input_file_name}/{try_count}', exist_ok=True)
        
        np.save(f'my_data/{input_file_name}/{try_count}/data.npy', data)
        np.save(f'my_data/{input_file_name}/{try_count}/label.npy', label)
        
    else:
        print('data_file is existed')

    data_creation_time = time.time()
    print('data_creation_time : ', datetime.timedelta(data_creation_time - start))

    # 学習結果のファイルを作成
    os.makedirs(name=f'pth_file/{output_file_name}', exist_ok=True)
    os.makedirs(f'results/{output_file_name}', exist_ok=True)

    graph_type = meta["graph_type"]
    in_channels = meta["in_channels"]
    t_kernel_size = meta["t_kernel_size"]
    node_num = meta["node_num"]
    E = meta["E"]
    has_bn = meta["has_bn"]
    # train
    pair_train(NUM_PAIR, NUM_CLASSES, NUM_EPOCH, BATCH_SIZE, graph_type, in_channels, t_kernel_size, node_num, E, has_bn, f'{input_file_name}/{try_count}', f'{output_file_name}/{try_count}')
        
# vote
pair_vote(meta, input_file_name, output_file_name)



#++++++++++++++++++++++++++++++++++++++++++++++++ img_result++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
os.makedirs(name='imgs', exist_ok=True)

# 試行回数ごとの学習リスト
try_train_acc_list  = np.empty((try_num, NUM_EPOCH))
try_train_loss_list = np.empty((try_num, NUM_EPOCH))
try_val_acc_list    = np.empty((try_num, NUM_EPOCH))
try_val_loss_list   = np.empty((try_num, NUM_EPOCH))

for try_count in range(1, try_num+1):
  
  # ペアごとの学習リスト
  pair_train_acc_list  = np.empty((NUM_PAIR, NUM_EPOCH))
  pair_train_loss_list = np.empty((NUM_PAIR, NUM_EPOCH))
  pair_val_acc_list   = np.empty((NUM_PAIR, NUM_EPOCH))
  pair_val_loss_list  = np.empty((NUM_PAIR, NUM_EPOCH))
  
  for one_pair in range(1, NUM_PAIR+1):
    train_acc_list  = np.load(f'results/{output_file_name}/{try_count}/{one_pair}/train_acc_list.npy')
    train_loss_list = np.load(f'results/{output_file_name}/{try_count}/{one_pair}/train_loss_list.npy')
    val_acc_list   = np.load(f'results/{output_file_name}/{try_count}/{one_pair}/val_acc_list.npy')
    val_loss_list  = np.load(f'results/{output_file_name}/{try_count}/{one_pair}/val_loss_list.npy')
    
    pair_train_acc_list[one_pair-1] = train_acc_list
    pair_train_loss_list[one_pair-1] = train_loss_list
    pair_val_acc_list[one_pair-1] = val_acc_list
    pair_val_loss_list[one_pair-1] = val_loss_list
    
  pair_train_acc_list = np.mean(pair_train_acc_list, axis=0)
  pair_train_loss_list = np.mean(pair_train_loss_list, axis=0)  
  pair_val_acc_list = np.mean(pair_val_acc_list, axis=0)
  pair_val_loss_list = np.mean(pair_val_loss_list, axis=0)
  
  
  try_train_acc_list[try_count-1] = pair_train_acc_list
  try_train_loss_list[try_count-1] = pair_train_loss_list
  try_val_acc_list[try_count-1] = pair_val_acc_list
  try_val_loss_list[try_count-1] = pair_val_loss_list
  
try_train_acc_list = np.mean(try_train_acc_list, axis=0)
try_train_loss_list = np.mean(try_train_loss_list, axis=0)  
try_val_acc_list = np.mean(try_val_acc_list, axis=0)
try_val_loss_list = np.mean(try_val_loss_list, axis=0)


os.makedirs(name=f'imgs/{output_file_name}', exist_ok=True)

plt.figure()
plt.plot(range(NUM_EPOCH), try_train_loss_list, color='blue', linestyle='-', label='train_loss')
plt.plot(range(NUM_EPOCH), try_val_loss_list, color='green', linestyle='--', label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.grid()
plt.savefig(f'imgs/{output_file_name}/loss.png')
plt.close()

plt.figure()
plt.plot(range(NUM_EPOCH), try_train_acc_list, color='blue', linestyle='-', label='train_acc')
plt.plot(range(NUM_EPOCH), try_val_acc_list, color='green', linestyle='--', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation accuracy')
plt.grid()
plt.savefig(f'imgs/{output_file_name}/acc.png')
plt.close()



confusion_matrix = np.zeros((3, 3))
vote_acc_list = np.empty((try_count))

for try_count in range(1, try_num+1):
  n_confusion_matrix  = np.load(f'results/{output_file_name}/{try_count}/conf_mat.npy')
  confusion_matrix += n_confusion_matrix
  # vote_acc_list[n-1] = np.load(f'results/{output_file_name}/vote_acc.npy')
  vote_acc_list[try_count-1] = np.load(f'results/{output_file_name}/{try_count}/vote_acc.npy')
  
mean = np.mean(vote_acc_list)
std = np.std(vote_acc_list)

sns.heatmap(confusion_matrix, annot=True, cmap='Blues',  fmt=".0f", cbar=False, annot_kws={"fontsize":18})
plt.title(f'average : {np.round(mean, decimals=2)}, std : {np.round(std, decimals=2)}')
plt.savefig(f'imgs/{output_file_name}/conf_mat.png')
print(confusion_matrix)