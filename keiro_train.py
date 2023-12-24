import numpy as np
import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
import argparse

from utils.train import pair_train, walk_path_train
from utils.vote import pair_vote, walk_path_vote
from utils.create_data import create_two_person_data
from utils.img_result import pair_img_result, walk_path_img_result

parser = argparse.ArgumentParser(description="Description of your program")
parser.add_argument("config_file_name", type=str, help="config_file_name")
args = parser.parse_args()
config_file_name = args.config_file_name


keiro = [["up", "down"], ["up", "lower_left"],  ["upper_right", "down"], ["upper_right", "lower_left"]]
keiro_index = [[0, 2], [0, 3], [1, 2], [1, 3]]


with open(f"configs/{config_file_name}.json", "r") as json_file:
    meta = json.load(json_file)


for i, k_n in enumerate(keiro):
    # 実験識別ネーム
    name = f'{meta["name"]}_{i}'
    
    meta["train_walk_num"] = keiro_index[i][0]
    meta["test_walk_num"] = keiro_index[i][1]
    
    node_option_name = meta["node_option_name"]
    print("実験名", name)
    print("ノードオプション", node_option_name if node_option_name != "" else "なし")

    num_people = meta["num_people"]
    node_index = meta["node_index"]

    # 学習パラメータ
    try_num = meta["try_num"]

    # 短動画パラメータ
    interval_num = meta["interval_num"]
    min_move_num = meta["min_move_num"]
    out_frame_num = meta["out_frame_num"]
    print("\ninterval_num", interval_num)
    print("min_move_num", min_move_num)
    print("out_frame_num", out_frame_num)


    # *****************変数定義*********************************************************************
    input_file_name = f"{node_option_name}"
    output_file_name = f"{name}_{node_option_name}"


    # ******************試行開始**********************************************************************
    for try_count in range(1, try_num + 1):
        os.makedirs(name=f"my_data/{input_file_name}", exist_ok=True)

        # ***********************************データ作成****************************************************************

        if num_people == 2:
            meta["left_node_index"] = node_index[0]
            meta["right_node_index"] = node_index[1]
            create_two_person_data(meta, try_count, input_file_name)

        # 学習結果のファイルを作成
        os.makedirs(name=f"pth_file/{output_file_name}", exist_ok=True)
        os.makedirs(f"results/{output_file_name}", exist_ok=True)

        # ******************************************* train ****************************************************************************************************
        if (meta["train_type"] == "pair"):
            pair_train(meta, try_count, f"{input_file_name}", f"{output_file_name}")
        elif (meta["train_type"] == "walk_path"):
            walk_path_train(meta, try_count, input_file_name, output_file_name)


    # *************************************************vote****************************************************************************************
    if (meta["train_type"] == "pair"):
        pair_vote(meta, input_file_name, output_file_name)
    elif (meta["train_type"] == "walk_path"):
        walk_path_vote(meta, input_file_name, output_file_name)



    # ++++++++++++++++++++++++++++++++++++++++++++++++ img_result++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (meta["train_type"] == "pair"):
        pair_img_result(meta, output_file_name)
    elif (meta["train_type"] == "walk_path"):
        walk_path_img_result(meta, output_file_name)


keiro_train_acc_list    = np.empty((4, meta["NUM_EPOCH"]))
keiro_train_loss_list   = np.empty((4, meta["NUM_EPOCH"]))
keiro_val_acc_list      = np.empty((4, meta["NUM_EPOCH"]))
keiro_val_loss_list      = np.empty((4, meta["NUM_EPOCH"]))

# for i, k_n in enumerate(keiro):
#     name = f'{meta["name"]}_{i}'
#     output_file_name = f"{name}_{node_option_name}"
    
#         # 学習パラメータ
#     try_num = meta["try_num"]
#     NUM_CLASSES = meta["NUM_CLASSES"]
#     NUM_EPOCH = meta["NUM_EPOCH"]
#     NUM_PAIR = meta["NUM_PAIR"]

#     # 短動画パラメータ
#     interval_num = meta["interval_num"]
#     min_move_num = meta["min_move_num"]
#     out_frame_num = meta["out_frame_num"]
#     print("\ninterval_num", interval_num)
#     print("min_move_num", min_move_num)
#     print("out_frame_num", out_frame_num)

#     # 試行回数ごとの学習リスト
#     try_train_acc_list = np.empty((try_num, NUM_EPOCH))
#     try_train_loss_list = np.empty((try_num, NUM_EPOCH))
#     try_val_acc_list = np.empty((try_num, NUM_EPOCH))
#     try_val_loss_list = np.empty((try_num, NUM_EPOCH))


#     for try_count in range(1, try_num + 1):


#         train_acc_list = np.load(f"results/{output_file_name}/{try_count}/train_acc_list.npy")
#         train_loss_list = np.load(f"results/{output_file_name}/{try_count}/train_loss_list.npy")
#         val_acc_list = np.load(f"results/{output_file_name}/{try_count}/val_acc_list.npy")
#         val_loss_list = np.load(f"results/{output_file_name}/{try_count}/val_loss_list.npy")

#         try_train_acc_list[try_count - 1] = train_acc_list
#         try_train_loss_list[try_count - 1] = train_loss_list
#         try_val_acc_list[try_count - 1] = val_acc_list
#         try_val_loss_list[try_count - 1] = val_loss_list

#     print(np.mean(try_train_acc_list, axis=0).shape)
#     keiro_train_acc_list[i - 1]  = np.mean(try_train_acc_list, axis=0)
#     keiro_train_loss_list[i - 1]   = np.mean(try_train_loss_list, axis=0)
#     keiro_val_acc_list[i - 1]      = np.mean(try_val_acc_list, axis=0)
#     keiro_val_loss_list[i - 1]     = np.mean(try_val_loss_list, axis=0)
     
# print(keiro_train_acc_list)
    
