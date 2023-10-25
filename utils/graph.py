import numpy as np
from .visualize import draw_adjancy_matrix

class Hop_Adjancey_Graph():
  def __init__(self, hop_size):
    self.get_edge()

    # hop: hop数分離れた関節を結びます.
    # 例えばhop=2だと, 手首は肘だけではなく肩にも繋がっています.
    self.hop_size = hop_size
    self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)

    # 隣接行列を作ります.ここではhop数ごとに隣接行列を作成します.
    # hopが2の時, 0hop, 1hop, 2hopの３つの隣接行列が作成されます.
    self.get_adjacency()

  def __str__(self):
    return self.A

  def get_edge(self):
    self.num_node = 17
    self_link = [(i, i) for i in range(self.num_node)] # ループ
    neighbor_base = [[0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 7], [6, 8],  [7, 9], [8, 10], [5,11],  [6, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
    self.edge = self_link + neighbor_link

  def get_adjacency(self):
    valid_hop = range(0, self.hop_size + 1, 1)
    adjacency = np.zeros((self.num_node, self.num_node))
    for hop in valid_hop:
        adjacency[self.hop_dis == hop] = 1
    normalize_adjacency = self.normalize_digraph(adjacency)
    A = np.zeros((len(valid_hop), self.num_node, self.num_node))
    for i, hop in enumerate(valid_hop):
        A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
    self.A = A

  def get_hop_distance(self, num_node, edge, hop_size):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(hop_size, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

  def normalize_digraph(self, A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    DAD = np.dot(A, Dn)
    # return DAD
    return DAD
  


class Base_Graph():
  def __init__(self, node_num, E):
    
    self.node_num = node_num
    self.E = E
    
    reversed_E = [[j, i] for [i, j] in E]
    I = [[i, i] for i in range(self.node_num)]
    new_E = E + reversed_E + I
    
    self.A = self.edge2mat(new_E)
    draw_adjancy_matrix(self.A[0], 'imgs', 'confusion_matrix')
    
  def __str__(self):
    return self.A

  def edge2mat(self, E):      
      A = np.zeros((1, self.node_num, self.node_num))
      for i, j in E:
          A[0, j, i] = 1
              
      D = self.get_D(A[0], pow=-0.5)
      A[0] = D @ A[0] @ D      
      return A

  def get_D(self, A, pow=-1):
      d_ii = np.sum(A, 0)
      D = np.zeros_like(A)
      for i in range(len(A)):
          D[i, i] = d_ii[i]**(pow)
      return D