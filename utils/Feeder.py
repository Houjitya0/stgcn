import torch
import numpy as np

# 訓練データを読み込み
class Pair_Train_Dataset(torch.utils.data.Dataset):
  def __init__(self, data_path, label_path, pair_num, NUM_PAIR):
      super().__init__()      
      tmp = np.load(data_path).astype(np.float32)
      N, T, V, C = tmp.shape
      self.data = np.concatenate((tmp[:(pair_num-1)*(N//NUM_PAIR)], tmp[(pair_num)*(N//NUM_PAIR):]))
      tmp = np.load(label_path)
      self.label = np.concatenate((tmp[:(pair_num-1)*(N//NUM_PAIR)], tmp[(pair_num)*(N//NUM_PAIR):]))
           
  def __len__(self):
      return len(self.label)

  def __iter__(self):
      return self

  def __getitem__(self, index):
      data = self.data[index]
      label = self.label[index]

      return data, label
  
  
# テストデータを読み込み
class Pair_Test_Dataset(torch.utils.data.Dataset):
  def __init__(self, data_path, label_path, pair_num, NUM_PAIR):
      super().__init__()      
      tmp = np.load(data_path).astype(np.float32)
      N, T, V, C = tmp.shape
      self.data = tmp[(pair_num-1)*(N//NUM_PAIR):(pair_num)*(N//NUM_PAIR)]
      tmp = np.load(label_path)
      self.label = tmp[(pair_num-1)*(N//NUM_PAIR):(pair_num)*(N//NUM_PAIR)]
           
  def __len__(self):
      return len(self.label)

  def __iter__(self):
      return self

  def __getitem__(self, index):
      data = self.data[index]
      label = self.label[index]

      return data, label
  
  
class Pair_Walk_Path_Train_Dataset(torch.utils.data.Dataset):
  def __init__(self, data_path, label_path, NUM_PAIR, NUM_CLASSES, walk_path_num):
      super().__init__()
      tmp = np.load(data_path).astype(np.float32)
      N, T, V, C = tmp.shape
      slash = N // 4
      self.data = tmp[(walk_path_num)*(slash): (walk_path_num+1)*(slash)]
      tmp = np.load(label_path)
      self.label = tmp[(walk_path_num)*(slash): (walk_path_num+1)*(slash)]
           
  def __len__(self):
      return len(self.label)

  def __iter__(self):
      return self

  def __getitem__(self, index):
      data = self.data[index]
      label = self.label[index]

      return data, label
  
  
class Pair_Walk_Path_Test_Dataset(torch.utils.data.Dataset):
  def __init__(self, data_path, label_path, NUM_PAIR, NUM_CLASSES, walk_path_num):
      super().__init__()
      tmp = np.load(data_path).astype(np.float32)
      N, T, V, C = tmp.shape
      slash = N // 4
      self.data = tmp[(walk_path_num)*(slash): (walk_path_num+1)*(slash)]
      tmp = np.load(label_path)
      self.label = tmp[(walk_path_num)*(slash): (walk_path_num+1)*(slash)]
           
  def __len__(self):
      return len(self.label)

  def __iter__(self):
      return self

  def __getitem__(self, index):
      data = self.data[index]
      label = self.label[index]

      return data, label