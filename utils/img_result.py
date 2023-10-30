
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def pair_img_result(meta, output_file_name):
    # 学習パラメータ
    try_num = meta["try_num"]
    NUM_CLASSES = meta["NUM_CLASSES"]
    NUM_EPOCH = meta["NUM_EPOCH"]
    NUM_PAIR = meta["NUM_PAIR"]

    # 短動画パラメータ
    interval_num = meta["interval_num"]
    min_move_num = meta["min_move_num"]
    out_frame_num = meta["out_frame_num"]
    print("\ninterval_num", interval_num)
    print("min_move_num", min_move_num)
    print("out_frame_num", out_frame_num)

    os.makedirs(name="imgs", exist_ok=True)

    # 試行回数ごとの学習リスト
    try_train_acc_list = np.empty((try_num, NUM_EPOCH))
    try_train_loss_list = np.empty((try_num, NUM_EPOCH))
    try_val_acc_list = np.empty((try_num, NUM_EPOCH))
    try_val_loss_list = np.empty((try_num, NUM_EPOCH))

    for try_count in range(1, try_num + 1):
        # ペアごとの学習リスト
        pair_train_acc_list = np.empty((NUM_PAIR, NUM_EPOCH))
        pair_train_loss_list = np.empty((NUM_PAIR, NUM_EPOCH))
        pair_val_acc_list = np.empty((NUM_PAIR, NUM_EPOCH))
        pair_val_loss_list = np.empty((NUM_PAIR, NUM_EPOCH))

        for one_pair in range(1, NUM_PAIR + 1):
            train_acc_list = np.load(f"results/{output_file_name}/{try_count}/{one_pair}/train_acc_list.npy")
            train_loss_list = np.load(f"results/{output_file_name}/{try_count}/{one_pair}/train_loss_list.npy")
            val_acc_list = np.load(f"results/{output_file_name}/{try_count}/{one_pair}/val_acc_list.npy")
            val_loss_list = np.load(f"results/{output_file_name}/{try_count}/{one_pair}/val_loss_list.npy")

            pair_train_acc_list[one_pair - 1] = train_acc_list
            pair_train_loss_list[one_pair - 1] = train_loss_list
            pair_val_acc_list[one_pair - 1] = val_acc_list
            pair_val_loss_list[one_pair - 1] = val_loss_list

        pair_train_acc_list = np.mean(pair_train_acc_list, axis=0)
        pair_train_loss_list = np.mean(pair_train_loss_list, axis=0)
        pair_val_acc_list = np.mean(pair_val_acc_list, axis=0)
        pair_val_loss_list = np.mean(pair_val_loss_list, axis=0)

        try_train_acc_list[try_count - 1] = pair_train_acc_list
        try_train_loss_list[try_count - 1] = pair_train_loss_list
        try_val_acc_list[try_count - 1] = pair_val_acc_list
        try_val_loss_list[try_count - 1] = pair_val_loss_list

    try_train_acc_list = np.mean(try_train_acc_list, axis=0)
    try_train_loss_list = np.mean(try_train_loss_list, axis=0)
    try_val_acc_list = np.mean(try_val_acc_list, axis=0)
    try_val_loss_list = np.mean(try_val_loss_list, axis=0)


    os.makedirs(name=f"imgs/{output_file_name}", exist_ok=True)

    plt.figure()
    plt.plot(
        range(NUM_EPOCH),
        try_train_loss_list,
        color="blue",
        linestyle="-",
        label="train_loss",
    )
    plt.plot(
        range(NUM_EPOCH), try_val_loss_list, color="green", linestyle="--", label="val_loss"
    )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and validation loss")
    plt.grid()
    plt.savefig(f"imgs/{output_file_name}/loss.png")
    plt.close()

    plt.figure()
    plt.plot(
        range(NUM_EPOCH), try_train_acc_list, color="blue", linestyle="-", label="train_acc"
    )
    plt.plot(
        range(NUM_EPOCH), try_val_acc_list, color="green", linestyle="--", label="val_acc"
    )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Training and validation accuracy")
    plt.grid()
    plt.savefig(f"imgs/{output_file_name}/acc.png")
    plt.close()


    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    vote_acc_list = np.empty((try_count))

    for try_count in range(1, try_num + 1):
        n_confusion_matrix = np.load(f"results/{output_file_name}/{try_count}/conf_mat.npy")
        confusion_matrix += n_confusion_matrix
        # vote_acc_list[n-1] = np.load(f'results/{output_file_name}/vote_acc.npy')
        vote_acc_list[try_count - 1] = np.load(f"results/{output_file_name}/{try_count}/vote_acc.npy")

    mean = np.mean(vote_acc_list)
    std = np.std(vote_acc_list)

    sns.heatmap(
        confusion_matrix,
        annot=True,
        cmap="Blues",
        fmt=".0f",
        cbar=False,
        annot_kws={"fontsize": 18},
    )
    plt.title(f"average : {np.round(mean, decimals=2)}, std : {np.round(std, decimals=2)}")
    plt.savefig(f"imgs/{output_file_name}/conf_mat.png")
    print(confusion_matrix)
    
    
def walk_path_img_result(meta, output_file_name):
    # 学習パラメータ
    try_num = meta["try_num"]
    NUM_CLASSES = meta["NUM_CLASSES"]
    NUM_EPOCH = meta["NUM_EPOCH"]
    NUM_PAIR = meta["NUM_PAIR"]

    # 短動画パラメータ
    interval_num = meta["interval_num"]
    min_move_num = meta["min_move_num"]
    out_frame_num = meta["out_frame_num"]
    print("\ninterval_num", interval_num)
    print("min_move_num", min_move_num)
    print("out_frame_num", out_frame_num)

    os.makedirs(name="imgs", exist_ok=True)

    # 試行回数ごとの学習リスト
    try_train_acc_list = np.empty((try_num, NUM_EPOCH))
    try_train_loss_list = np.empty((try_num, NUM_EPOCH))
    try_val_acc_list = np.empty((try_num, NUM_EPOCH))
    try_val_loss_list = np.empty((try_num, NUM_EPOCH))

    for try_count in range(1, try_num + 1):


        train_acc_list = np.load(f"results/{output_file_name}/{try_count}/train_acc_list.npy")
        train_loss_list = np.load(f"results/{output_file_name}/{try_count}/train_loss_list.npy")
        val_acc_list = np.load(f"results/{output_file_name}/{try_count}/val_acc_list.npy")
        val_loss_list = np.load(f"results/{output_file_name}/{try_count}/val_loss_list.npy")

        try_train_acc_list[try_count - 1] = train_acc_list
        try_train_loss_list[try_count - 1] = train_loss_list
        try_val_acc_list[try_count - 1] = val_acc_list
        try_val_loss_list[try_count - 1] = val_loss_list

    try_train_acc_list = np.mean(try_train_acc_list, axis=0)
    try_train_loss_list = np.mean(try_train_loss_list, axis=0)
    try_val_acc_list = np.mean(try_val_acc_list, axis=0)
    try_val_loss_list = np.mean(try_val_loss_list, axis=0)


    os.makedirs(name=f"imgs/{output_file_name}", exist_ok=True)

    plt.figure()
    plt.plot(
        range(NUM_EPOCH),
        try_train_loss_list,
        color="blue",
        linestyle="-",
        label="train_loss",
    )
    plt.plot(
        range(NUM_EPOCH), try_val_loss_list, color="green", linestyle="--", label="val_loss"
    )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and validation loss")
    plt.grid()
    plt.savefig(f"imgs/{output_file_name}/loss.png")
    plt.close()

    plt.figure()
    plt.plot(
        range(NUM_EPOCH), try_train_acc_list, color="blue", linestyle="-", label="train_acc"
    )
    plt.plot(
        range(NUM_EPOCH), try_val_acc_list, color="green", linestyle="--", label="val_acc"
    )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Training and validation accuracy")
    plt.grid()
    plt.savefig(f"imgs/{output_file_name}/acc.png")
    plt.close()


    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    vote_acc_list = np.empty((try_count))

    for try_count in range(1, try_num + 1):
        n_confusion_matrix = np.load(f"results/{output_file_name}/{try_count}/conf_mat.npy")
        confusion_matrix += n_confusion_matrix
        # vote_acc_list[n-1] = np.load(f'results/{output_file_name}/vote_acc.npy')
        vote_acc_list[try_count - 1] = np.load(f"results/{output_file_name}/{try_count}/vote_acc.npy")

    mean = np.mean(vote_acc_list)
    std = np.std(vote_acc_list)

    sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt=".0f", cbar=False, annot_kws={"fontsize": 18})
    plt.title(f"average : {np.round(mean, decimals=2)}, std : {np.round(std, decimals=2)}")
    plt.savefig(f"imgs/{output_file_name}/conf_mat.png")
    print(confusion_matrix)
    