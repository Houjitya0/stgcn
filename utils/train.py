import torch
import os
import numpy as np
import datetime
import time

from utils.stgcn import ST_GCN
from utils.Feeder import Pair_Test_Dataset, Pair_Train_Dataset, Pair_Walk_Path_Train_Dataset, Pair_Walk_Path_Test_Dataset

def pair_train(meta, try_count, input_file_name, output_file_name):
  
    print(f"試行回数 : {try_count}目")
    start_time = time.time()
    
    NUM_PAIR = meta["NUM_PAIR"]
    NUM_CLASSES = meta["NUM_CLASSES"]
    NUM_EPOCH = meta["NUM_EPOCH"]
    BATCH_SIZE = meta["BATCH_SIZE"] 
    graph_type = meta["graph_type"]
    in_channels = meta["in_channels"] 
    t_kernel_size = meta["t_kernel_size"]
    node_num = meta["node_num"]
    E = meta["E"]
    has_bn = meta["has_bn"]

    # Pytorch
    seed=123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    device = torch.device(0)

    os.makedirs(f'pth_file/{output_file_name}/{try_count}', exist_ok=True)

    # **************************深層学習**********************************************************
    for one_pair in range(1, NUM_PAIR+1):
        data_loader = dict()
        data_loader['train'] = torch.utils.data.DataLoader(dataset=Pair_Train_Dataset(data_path=f'my_data/{input_file_name}/{try_count}/data.npy', label_path=f'my_data/{input_file_name}/{try_count}/label.npy', pair_num=one_pair, NUM_PAIR=NUM_PAIR), batch_size=BATCH_SIZE, shuffle=True)
        data_loader['test'] = torch.utils.data.DataLoader(dataset=Pair_Test_Dataset(data_path=f'my_data/{input_file_name}/{try_count}/data.npy', label_path=f'my_data/{input_file_name}/{try_count}/label.npy', pair_num=one_pair, NUM_PAIR=NUM_PAIR), batch_size=BATCH_SIZE, shuffle=False)
        
        print('one_pair', one_pair)
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        
        # 組ループの途中で止まってしまった場合の処理
        if(os.path.exists(f'pth_file/{output_file_name}/{try_count}/model_weight{one_pair}')) and (os.path.exists(f'results/{output_file_name}/{try_count}/{one_pair}')):
            print(one_pair, 'yes')
            continue    
        
        model = ST_GCN(graph_type=graph_type, NUM_CLASSES=NUM_CLASSES, in_channels=in_channels, t_kernel_size=t_kernel_size, node_num=node_num, E=E, has_bn=has_bn).cuda()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        # 学習
        print('\nStart training')
        for epoch in range(NUM_EPOCH):
            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0
            
            #train
            model.train()
            for i, (data, label) in enumerate(data_loader['train']):
                        
                data = data.to(device)
                label = label.to(device)      
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, label)
                train_loss += loss.item()
                train_acc += (output.max(1)[1] == label).sum().item()
                loss.backward()
                optimizer.step()
            
            avg_train_loss = train_loss / len(data_loader['train'].dataset)
            avg_train_acc = train_acc / len(data_loader['train'].dataset)
            
            #val
            model.eval()
            with torch.no_grad():
                for data, label in data_loader['test']:
                    data = data.to(device)
                    label = label.to(device)
                    output = model(data)
                    loss = criterion(output, label)
                    val_loss += loss.item()
                    val_acc += (output.max(1)[1] == label).sum().item()
            avg_val_loss = val_loss / len(data_loader['test'].dataset)
            avg_val_acc = val_acc / len(data_loader['test'].dataset)
            
            print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' 
                        .format(epoch+1, NUM_EPOCH, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
            train_loss_list.append(avg_train_loss)
            train_acc_list.append(avg_train_acc)
            val_loss_list.append(avg_val_loss)
            val_acc_list.append(avg_val_acc)


        # 1pairごとに学習記録を保存
        print(f"{output_file_name}/{try_count}")
        os.makedirs(f'results/{output_file_name}/{try_count}/{one_pair}', exist_ok=True)
            
        np.save(f'results/{output_file_name}/{try_count}/{one_pair}/train_loss_list', train_loss_list)
        np.save(f'results/{output_file_name}/{try_count}/{one_pair}/train_acc_list', train_acc_list)
        np.save(f'results/{output_file_name}/{try_count}/{one_pair}/val_loss_list', val_loss_list)
        np.save(f'results/{output_file_name}/{try_count}/{one_pair}/val_acc_list', val_acc_list)
        torch.save(model, f'pth_file/{output_file_name}/{try_count}/model_weight{one_pair}')
        del model
        
        
    print('learn_time', datetime.timedelta(seconds= time.time() - start_time))




def walk_path_train(meta, try_count, input_file_name, output_file_name):
    
    
    start_time = time.time()

    NUM_CLASSES = meta["NUM_CLASSES"]
    NUM_PAIR = meta["NUM_PAIR"]
    NUM_EPOCH = meta["NUM_EPOCH"]
    BATCH_SIZE = meta["BATCH_SIZE"]
    train_walk_num = meta["train_walk_num"] 
    test_walk_num = meta["test_walk_num"]
    graph_type = meta["graph_type"]
    in_channels = meta["in_channels"] 
    t_kernel_size = meta["t_kernel_size"]
    node_num = meta["node_num"]
    E = meta["E"]
    has_bn = meta["has_bn"]

    # Pytorch
    seed=123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    device = torch.device(0)

    os.makedirs(f'pth_file/{output_file_name}/{try_count}', exist_ok=True)
    
    
    data_loader = dict()
    
   
    data_loader['train'] = torch.utils.data.DataLoader(dataset=Pair_Walk_Path_Train_Dataset(data_path=f'my_data/{input_file_name}/{try_count}/data.npy', label_path=f'my_data/{input_file_name}/{try_count}/label.npy', NUM_PAIR=NUM_PAIR,NUM_CLASSES=NUM_CLASSES,walk_path_num=train_walk_num), batch_size=BATCH_SIZE, shuffle=True)
    data_loader['test'] = torch.utils.data.DataLoader(dataset=Pair_Walk_Path_Train_Dataset(data_path=f'my_data/{input_file_name}/{try_count}/data.npy', label_path=f'my_data/{input_file_name}/{try_count}/label.npy', NUM_PAIR=NUM_PAIR,NUM_CLASSES=NUM_CLASSES,walk_path_num=test_walk_num), batch_size=BATCH_SIZE, shuffle=False)
    
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []


    # 組ループの途中で止まってしまった場合の処理
    if(os.path.exists(f'pth_file/{output_file_name}/{try_count}/model_weight')) and (os.path.exists(f'results/{output_file_name}/{try_count}')):
        print(try_count, 'yes')
        return
        
    model = ST_GCN(graph_type=graph_type, NUM_CLASSES=NUM_CLASSES, in_channels=in_channels, t_kernel_size=t_kernel_size, node_num=node_num, E=E, has_bn=has_bn).cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # 学習
    print('\nStart training')
    for epoch in range(NUM_EPOCH):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        
        #train
        model.train()
        for i, (data, label) in enumerate(data_loader['train']):
                    
            data = data.to(device)
            label = label.to(device)      
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            train_loss += loss.item()
            train_acc += (output.max(1)[1] == label).sum().item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(data_loader['train'].dataset)
        avg_train_acc = train_acc / len(data_loader['train'].dataset)
        
        #val
        model.eval()
        with torch.no_grad():
            for data, label in data_loader['test']:
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = criterion(output, label)
                val_loss += loss.item()
                val_acc += (output.max(1)[1] == label).sum().item()
        avg_val_loss = val_loss / len(data_loader['test'].dataset)
        avg_val_acc = val_acc / len(data_loader['test'].dataset)
        
        print ('Epoch [{}/{}], train_loss: {train_loss:.4f}, train_acc : {train_acc} ,val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' 
                    .format(epoch+1, NUM_EPOCH, i+1, train_loss=avg_train_loss, train_acc=avg_train_acc, val_loss=avg_val_loss, val_acc=avg_val_acc))
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)


    print(f"{output_file_name}/{try_count}")
    os.makedirs(f'results/{output_file_name}/{try_count}', exist_ok=True)
        
    np.save(f'results/{output_file_name}/{try_count}/train_loss_list', train_loss_list)
    np.save(f'results/{output_file_name}/{try_count}/train_acc_list', train_acc_list)
    np.save(f'results/{output_file_name}/{try_count}/val_loss_list', val_loss_list)
    np.save(f'results/{output_file_name}/{try_count}/val_acc_list', val_acc_list)
    torch.save(model, f'pth_file/{output_file_name}/{try_count}/model_weight')
    del model
    
    print('learn_time', datetime.timedelta(seconds= time.time() - start_time))
