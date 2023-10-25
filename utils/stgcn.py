import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph import Base_Graph

class SpatialGraphConvolution(nn.Module):
  def __init__(self, in_channels, out_channels, s_kernel_size):
    super().__init__()
    self.s_kernel_size = s_kernel_size
    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels * s_kernel_size,
                          kernel_size=1)

  def forward(self, x, A):
    x = self.conv(x)
    n, kc, t, v = x.size()
    x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)

    x = torch.einsum('nkctv,kvw->nctw', (x, A))
    return x.contiguous()


class STGC_block(nn.Module):
  def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.5):
    super().__init__()
    # 空間グラフの畳み込み
    self.sgc = SpatialGraphConvolution(in_channels=in_channels,
                                       out_channels=out_channels,
                                       s_kernel_size=A_size[0])

    # Learnable weight matrix M エッジに重みを与えます. どのエッジが重要かを学習します.
    self.M = nn.Parameter(torch.ones(A_size))

    self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Conv2d(out_channels,
                                      out_channels,
                                      (t_kernel_size, 1),
                                      (stride, 1),
                                      ((t_kernel_size - 1) // 2, 0)),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())

  def forward(self, x, A):
    x = self.tgc(self.sgc(x, A * self.M))
    return x

class ST_GCN(nn.Module):
  def __init__(self, graph_type, NUM_CLASSES, in_channels, t_kernel_size, node_num, E, has_bn):
    super().__init__()
    # グラフ作成
    self.has_bn = has_bn
    self.E = E
    self.node_num = node_num
    graph = Base_Graph(self.node_num, self.E)
    A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
    self.register_buffer('A', A)
    A_size = A.size()

    if (self.has_bn):
      self.bn = nn.BatchNorm1d(in_channels * A_size[1])

    # STGC_blocks
    self.stgc1 = STGC_block(in_channels, 64, 1, t_kernel_size, A_size)
    self.stgc2 = STGC_block(64, 64, 1, t_kernel_size, A_size)
    self.stgc3 = STGC_block(64, 64, 1, t_kernel_size, A_size)
    self.stgc4 = STGC_block(64, 128, 2, t_kernel_size, A_size)
    self.stgc5 = STGC_block(128, 128, 1, t_kernel_size, A_size)
    self.stgc6 = STGC_block(128, 128, 1, t_kernel_size, A_size)
    self.stgc7 = STGC_block(128, 256, 2, t_kernel_size, A_size)
    self.stgc8 = STGC_block(256, 256, 1, t_kernel_size, A_size)
    self.stgc9 = STGC_block(256, 256, 1, t_kernel_size, A_size)

    # Prediction
    self.fc = nn.Linear(in_features=256, out_features=3)

  def forward(self, x):
     # Batch Normalization
    N, T, V, C = x.size() # batch, channel, frame, node
    x = x.permute(0, 2, 3, 1).contiguous().view(N, V * C, T)
    
    if (self.has_bn):
      x = self.bn(x)
      
    x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
    x = self.stgc1(x, self.A)
    x = self.stgc2(x, self.A) + x
    x = self.stgc3(x, self.A) + x
    x = self.stgc4(x, self.A)
    x = self.stgc5(x, self.A) + x
    x = self.stgc6(x, self.A) + x
    x = self.stgc7(x, self.A)
    x = self.stgc8(x, self.A) + x
    x = self.stgc9(x, self.A) + x
    
    x = F.avg_pool2d(x, x.size()[2:])
    x = x.view(N, -1, 1, 1)
    n, c, a, b = x.shape
    x = x.view(n, c)
    x = self.fc(x)
    x = x.view(x.size(0), -1)
    return x