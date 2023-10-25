import torch
import numpy as np

# 訓練データを読み込み
class Pair_Train_Dataset(torch.utils.data.Dataset):
  def __init__(self, data_path, label_path, pair_num):
      super().__init__()
      tmp = np.load(data_path).astype(np.float32)
      N, T, V, C = tmp.shape
      self.data = np.concatenate((tmp[:(pair_num-1)*(N//52)], tmp[(pair_num)*(N//52):]))
      tmp = np.load(label_path)
      self.label = np.concatenate((tmp[:(pair_num-1)*(N//52)], tmp[(pair_num)*(N//52):]))
           
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
  def __init__(self, data_path, label_path, pair_num):
      super().__init__()
      tmp = np.load(data_path).astype(np.float32)
      N, T, V, C = tmp.shape
      self.data = tmp[(pair_num-1)*(N//52) : pair_num*(N//52)]
      self.label = np.load(label_path)[(pair_num-1)*(N//52) : pair_num*(N//52)]
           
  def __len__(self):
      return len(self.label)

  def __iter__(self):
      return self

  def __getitem__(self, index):
      data = self.data[index]
      label = self.label[index]

      return data, label