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