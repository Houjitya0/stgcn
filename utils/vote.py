import numpy as np
import torch
from collections import Counter
from utils.Feeder import Pair_Test_Dataset, Pair_Walk_Path_Train_Dataset

def pair_vote(meta, input_file_name, output_file_name):

    seed=123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    device = torch.device(0)
    
    try_num = meta["try_num"]
    NUM_PAIR = meta["NUM_PAIR"]
    BATCH_SIZE = meta["BATCH_SIZE"]
    min_move_num = meta["min_move_num"]
    num_people = meta["num_people"]

    for try_count in range(1, try_num+1):

        # 動画ごとに投票を行う
        correct = 0
        confusion_matrix = np.zeros((3, 3))
        predicts = []
        
        for one_pair in range(1, NUM_PAIR+1):
            # 学習済みモデルの読み込みを行い、評価モードに
            model = torch.load(f'pth_file/{output_file_name}/{try_count}/model_weight{one_pair}')
            model.eval()
            # 学習データの読み込み
            data_loader = dict()
            data_loader['test'] = torch.utils.data.DataLoader(dataset=Pair_Test_Dataset(data_path=f'my_data/{input_file_name}/{try_count}/data.npy', label_path=f'my_data/{input_file_name}/{try_count}/label.npy', pair_num=one_pair, NUM_PAIR=NUM_PAIR), batch_size=BATCH_SIZE, shuffle=False)
            
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(data_loader['test']):    
                    data = data.cuda()
                    label = label.cuda()
                    output = model(data)
                    _, predict = torch.max(output.data, 1)            
                    predict = predict.to('cpu').detach().numpy().copy()
                    predicts += predict.tolist()
            
        print('p', len(predicts))
        # ペア数x歩行経路x活発さラベル
        VIDEO_NUM = 52 * 4 * 3

        if (num_people == 2):
            graph_per_video = 1
        elif (num_people ==1):
            graph_per_video = 2
            

        for i in range(VIDEO_NUM): 
            # 投票
            a = predicts[(i)*(graph_per_video*min_move_num):(i+1)*(graph_per_video*min_move_num)]
            count = Counter(a)
            most_common_value = count.most_common(1)[0][0]

            confusion_matrix[i%3, most_common_value] += 1

            if most_common_value == (i % 3):
                correct += 1
                    
        vote_acc =  100. * correct / (VIDEO_NUM*graph_per_video)

        np.save(f"results/{output_file_name}/{try_count}/conf_mat", confusion_matrix)
        np.save(f"results/{output_file_name}/{try_count}/vote_acc", vote_acc)

        print(confusion_matrix)
        print('# vote : ', vote_acc)
        
        
        
def walk_path_vote(meta, input_file_name, output_file_name):
    seed=123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    device = torch.device(0)
    
    try_num = meta["try_num"]
    test_walk_num = meta["test_walk_num"]
    NUM_CLASSES = meta["NUM_CLASSES"]
    NUM_PAIR = meta["NUM_PAIR"]
    BATCH_SIZE = meta["BATCH_SIZE"]
    min_move_num = meta["min_move_num"]
    num_people = meta["num_people"]
    predicts = []
    
    for try_count in range(1, try_num+1):

        # 動画ごとに投票を行う
        correct = 0
        confusion_matrix = np.zeros((3, 3))
        
        # 学習済みモデルの読み込みを行い、評価モードに
        model = torch.load(f'pth_file/{output_file_name}/{try_count}/model_weight')
        model.eval()
        # 学習データの読み込み
        data_loader = dict()
        data_loader['test'] = torch.utils.data.DataLoader(dataset=Pair_Walk_Path_Train_Dataset(data_path=f'my_data/{input_file_name}/{try_count}/data.npy', label_path=f'my_data/{input_file_name}/{try_count}/label.npy', NUM_PAIR=NUM_PAIR,NUM_CLASSES=NUM_CLASSES,walk_path_num=test_walk_num), batch_size=BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(data_loader['test']):    
                data = data.cuda()
                label = label.cuda()
                output = model(data)
                _, predict = torch.max(output.data, 1)            
                predict = predict.to('cpu').detach().numpy().copy()
                predicts += predict.tolist() 
                
        print("len_predict", predicts)
        
        if (num_people == 2):
            graph_per_video = 1
        elif (num_people ==1):
            graph_per_video = 2
            

        for i in range(NUM_CLASSES*NUM_PAIR): 
            # 投票
            a = predicts[(i)*(graph_per_video*min_move_num):(i+1)*(graph_per_video*min_move_num)]
            count = Counter(a)
            most_common_value = count.most_common(1)[0][0]

            confusion_matrix[i%3, most_common_value] += 1

            if most_common_value == (i % 3):
                correct += 1
                    
        vote_acc =  100. * correct / (NUM_CLASSES*NUM_PAIR*graph_per_video)

        np.save(f"results/{output_file_name}/{try_count}/conf_mat", confusion_matrix)
        np.save(f"results/{output_file_name}/{try_count}/vote_acc", vote_acc)

        print(confusion_matrix)
        print('# vote : ', vote_acc)