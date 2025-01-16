class InteractionAttention(nn.Module):
      r"""InterHand Temporal Fusion and Interaction Attention
    """
    def __init__(self, channels, num_heads=4, dropout=0.1):
        super(InteractionAttention, self).__init__()
        self.num_heads = num_heads
        self.channels_per_head = channels // num_heads
        self.qkv_left = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.qkv_right = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # InterHand Temporal Fusion
        N, C, T, V = x.size()

        # QKV from left hand
        x_left = x[:, :, :, :21]
        QKV_left = self.qkv_left(x_left)
        Q_left, K_left, V_left = torch.chunk(QKV_left, 3, dim=1)

        # QKV from right hand
        x_right = x[:, :, :, 21:]
        QKV_right = self.qkv_right(x_right)
        Q_right, K_right, V_right = torch.chunk(QKV_right, 3, dim=1)

        # reshape data in time dimension
        Q_left_time = Q_left.view(N, self.num_heads, self.channels_per_head, 21, T)
        K_right_time = K_right.view(N, self.num_heads, self.channels_per_head, 21, T)
        V_right_time = V_right.view(N, self.num_heads, self.channels_per_head, 21, T)
        Q_right_time = Q_right.view(N, self.num_heads, self.channels_per_head, 21, T)
        K_left_time = K_left.view(N, self.num_heads, self.channels_per_head, 21, T)
        V_left_time = V_left.view(N, self.num_heads, self.channels_per_head, 21, T)

        # Interhand Temporal Fusion for left hand
        d_k_time = Q_left_time.size(2)
        attn_time_left = torch.matmul(Q_left_time.permute(0, 1, 3, 4, 2), K_right_time.permute(0, 1, 3, 2, 4)) / (d_k_time ** 0.5)
        attn_time_left = F.softmax(attn_time_left, dim=-1)
        out_time_left = torch.matmul(attn_time_left, V_right_time.permute(0, 1, 3, 4, 2))
        out_time_left = out_time_left.permute(0, 1, 4, 2, 3).contiguous().view(N, -1, T, 21)

        # InterHand Temporal Fusion for right hand
        d_k_time = Q_right_time.size(2)
        attn_time_right = torch.matmul(Q_right_time.permute(0, 1, 3, 4, 2), K_left_time.permute(0, 1, 3, 2, 4)) / (d_k_time ** 0.5)
        attn_time_right = F.softmax(attn_time_right, dim=-1)
        out_time_right = torch.matmul(attn_time_right, V_left_time.permute(0, 1, 3, 4, 2))
        out_time_right = out_time_right.permute(0, 1, 4, 2, 3).contiguous().view(N, -1, T, 21)

        # concat two hand feature
        out_temporal = torch.cat([out_time_left, out_time_right], dim=3)

        # feature extractor
        out = out_temporal.view(N, C, T, V)
        residual = out
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.dropout(out)
        out = out + self.residual(residual + x)
        x = out

        # Interaction Attention
          
        # QKV from left hand
        x_left = x[:, :, :, :21]
        QKV_left = self.qkv_left(x_left)
        Q_left, K_left, V_left = torch.chunk(QKV_left, 3, dim=1)

        # QKV from right hand
        x_right = x[:, :, :, 21:]
        QKV_right = self.qkv_right(x_right)
        Q_right, K_right, V_right = torch.chunk(QKV_right, 3, dim=1)
        
        Q_left = Q_left.view(N, self.num_heads, self.channels_per_head, T, 21)
        K_left = K_left.view(N, self.num_heads, self.channels_per_head, T, 21)
        V_left = V_left.view(N, self.num_heads, self.channels_per_head, T, 21)
        Q_right = Q_right.view(N, self.num_heads, self.channels_per_head, T, 21)
        K_right = K_right.view(N, self.num_heads, self.channels_per_head, T, 21)
        V_right = V_right.view(N, self.num_heads, self.channels_per_head, T, 21)
        
        # Interaction Attention for left hand
        d_k_left = Q_right.size(2)
        # Q_right, K_left
        attn_right_to_left = torch.matmul(Q_right.view(N, self.num_heads, T * 21, -1), K_left.view(N, self.num_heads, T * 21, -1).transpose(-2, -1)) / (d_k_left ** 0.5)
        attn_right_to_left = F.softmax(attn_right_to_left, dim=-1)
        # V_left
        out_right_to_left = torch.matmul(attn_right_to_left, V_left.view(N, self.num_heads, T * 21, -1))
        out_right_to_left = out_right_to_left.view(N, self.num_heads, T, 21, -1).permute(0, 1, 4, 2, 3).contiguous().view(N, -1, T, 21)

        # Interaction Attention for right hand
        d_k_right = Q_left.size(2)
        # Q_left, K_right
        attn_left_to_right = torch.matmul(Q_left.view(N, self.num_heads, T * 21, -1), K_right.view(N, self.num_heads, T * 21, -1).transpose(-2, -1)) / (d_k_right ** 0.5)
        attn_left_to_right = F.softmax(attn_left_to_right, dim=-1)
        # V_right
        out_left_to_right = torch.matmul(attn_left_to_right, V_right.view(N, self.num_heads, T * 21, -1))
        out_left_to_right = out_left_to_right.view(N, self.num_heads, T, 21, -1).permute(0, 1, 4, 2, 3).contiguous().view(N, -1, T, 21)

        
        out = torch.cat([out_left_to_right, out_right_to_left], dim=3) 
        out = out.view(N, C, T, V)

        # feauture extractor
        residual = out
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.dropout(out)
        out = out + self.residual(residual + x)
        
        return out

class Graph_base():
    def __init__(self, hop_size):
        self.get_edge()
        self.hop_size = hop_size
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)
        self.get_adjacency()
    def __str__(self):
        return self.A
          
    # Structure of graph
    def get_edge(self):
        self.num_node = 42
        self_link = [(i, i) for i in range(self.num_node)] 
        neighbor_base = [(1,2),(2,3),(3,4),(4,5),(1,6),(6,7),(7,8),(8,9),(1,10),(10,11),
                             (11,12),(12,13),(1,14),(14,15),(15,16),(16,17),(1,18),(18,19),(19,20),(20,21),
                             (22,23),(23,24),(24,25),(25,26),(22,27),(27,28),(28,29),(29,30),(22,31),(31,32),
                             (32,33),(33,34),(22,35),(35,36),(36,37),(37,38),(22,39),(39,40),(40,41),(41,42)]
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
        return DAD

# Interaction Graph with distance
class Graph_distance():
    def __init__(self, hop_size):
        self.get_edge()
        self.hop_size = hop_size
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)
        self.get_adjacency()
    def __str__(self):
        return self.A

    # Edge for Interaction Graph
    def get_edge(self):
        self.num_node = 42
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_base = [(1,22),(2,23),(3,24),(4,25),(5,26),(6,27),(7,28),(8,29),(9,30),(10,31),
                             (11,32),(12,33),(13,34),(14,35),(15,36),(16,37),(17,38),(18,39),(19,40),(20,41),(21,42)]
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
        return DAD

class SpatialGraphConvolution_base(nn.Module):
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

class SpatialGraphConvolution_distance(nn.Module):
    r"""SpatialGraphConvolution with Interaction Graph.

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Input[2]: Input interaction graph adjacency matrix in :math:`(K, V, V)` format
        - Input[3]: Input distance matrix in :math:`(V, V)` format

    """
      
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels * s_kernel_size,
                          kernel_size=1)
          
    def forward(self, x, A, A1, D):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
    
        # Interaction Graph
        x1 = torch.einsum('nkctv,kvw->nctw', (x, A1)).contiguous()
        # Normal Graph
        x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()

        n, c, t, w = x1.size()
        x1 = x1.reshape(n, -1, t, w)
        n1, c1, t1, w1 = x1.size()
        x1 = x1.permute(0, 2, 1, 3).contiguous().view(n1 * t1, c1, w1)

        # add distance feature into x
        x1 = torch.bmm(x1, D)
        x1 = x1.reshape(x1.size())
        x1 = x1.view(n, t, c, w)
        x1 = x1.permute(0, 2, 1, 3).contiguous()
        x = x + x1
        return x.contiguous()

class STGC_block_base(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t0_kernel_size, t1_kernel_size, A0_size, dropout=0.5):
        super().__init__()

        self.sgc0 = SpatialGraphConvolution_base(in_channels=in_channels,
                                       out_channels=out_channels,
                                       s_kernel_size=A0_size[0])

        self.M0 = nn.Parameter(torch.ones(A0_size))
        self.M0t = nn.Parameter(torch.ones(A0_size))

        self.tgc0 = nn.Sequential(nn.BatchNorm2d(out_channels),
                            nn.ReLU(), 
                            nn.Dropout(dropout),
                            nn.Conv2d(out_channels,
                                      out_channels,
                                      (t0_kernel_size, 1),
                                      (stride, 1),
                                      ((t0_kernel_size - 1) // 2, 0)),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())
          
        self.tgc1 = nn.Sequential(nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Conv2d(out_channels,
                                      out_channels,
                                      (t1_kernel_size, 1),
                                      (stride, 1),
                                      ((t1_kernel_size - 1) // 2, 0)),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())

        # InterHand Temporal Fusion + Interaction Attention
        self.attn = InteractionAttention(out_channels)
    
    def forward(self, x, A0_base):
          
        x0 = self.tgc0(self.sgc0(x, A0_base * self.M0))
        x1 = self.tgc1(self.sgc0(x, A0_base * self.M0t))
        x0 = x0 + x1

        # InterHand Temporal Fusion + Interaction Attention
        x = self.attn(x0) + x0
        return x

class STGC_block_distance(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t0_kernel_size, t1_kernel_size, A0_size, dropout=0.5):
        super().__init__()
    
        self.sgc0 = SpatialGraphConvolution_distance(in_channels=in_channels,
                                       out_channels=out_channels,
                                       s_kernel_size=A0_size[0])
        self.M00 = nn.Parameter(torch.ones(A0_size))
        self.M01 = nn.Parameter(torch.ones(A0_size))
        self.M00t = nn.Parameter(torch.ones(A0_size))
        self.M01t = nn.Parameter(torch.ones(A0_size))
    
        self.tgc0 = nn.Sequential(nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Conv2d(out_channels,
                                      out_channels,
                                      (t0_kernel_size, 1),
                                      (stride, 1),
                                      ((t0_kernel_size - 1) // 2, 0)),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())
        self.tgc1 = nn.Sequential(nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Conv2d(out_channels,
                                      out_channels,
                                      (t1_kernel_size, 1),
                                      (stride, 1),
                                      ((t1_kernel_size - 1) // 2, 0)),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())

        # InterHand Temporal Fusion + Interaction Attention
        self.attn = InteractionAttention(out_channels)

    def forward(self, x, A0_base, A0_distance, D):    
    
        x0 = self.tgc0(self.sgc0(x, A0_base * self.M00, A0_distance * self.M01, D))
        x1 = self.tgc1(self.sgc0(x, A0_base * self.M00t, A0_distance * self.M01t, D))
        x0 = x0 + x1
   
        # InterHand Temporal Fusion + Interaction Attention
        x = self.attn(x0) + x0
        return x

class ST_GCN(nn.Module):
    def __init__(self, num_classes, num_views, in_channels, t0_kernel_size, t1_kernel_size, t2_kernel_size, hop0_size, hop1_size, hop2_size):
        super().__init__()

        graph0 = Graph_base(hop0_size)
        A0_base = torch.tensor(graph0.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A0_base', A0_base)
    
        graph1 = Graph_distance(hop0_size)
        A0_distance = torch.tensor(graph1.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A0_distance', A0_distance)
    
        A0_size = A0_base.size()
    
        graph2 = Graph_base(hop1_size)
        A1_base = torch.tensor(graph2.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A1_base', A1_base)
    
        graph3 = Graph_distance(hop1_size)
        A1_distance = torch.tensor(graph3.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A1_distance', A1_distance)
    
        A1_size = A1_base.size()
    
        graph4 = Graph_base(hop2_size)
        A2_base = torch.tensor(graph4.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A2_base', A2_base)
    
        graph5 = Graph_distance(hop2_size)
        A2_distance = torch.tensor(graph5.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A2_distance', A2_distance)
    
        A2_size = A2_base.size()
    
        self.bn = nn.BatchNorm1d((in_channels) * A0_size[1])

        self.stgc1 = STGC_block_distance(in_channels, 32, 1, t0_kernel_size, t1_kernel_size, A0_size)
        self.stgc2 = STGC_block_distance(32, 32, 1, t0_kernel_size, t1_kernel_size, A0_size)
        self.stgc3 = STGC_block_distance(32, 32, 1, t0_kernel_size, t1_kernel_size, A0_size)
        self.stgc4 = STGC_block_base(32, 64, 2, t0_kernel_size, t1_kernel_size, A0_size)
        self.stgc5 = STGC_block_base(64, 64, 1, t0_kernel_size, t1_kernel_size, A0_size)
        self.stgc6 = STGC_block_base(64,  64, 1, t0_kernel_size, t1_kernel_size, A0_size)
    
        # Prediction
        self.fc = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x, view_indices, mode='train'):

         N, C, T, V = x.size() # batch, channel, frame, node

        # calculate distance matrix 
        D = x.contiguous().view(N * T, V, C)
        distances = ((D.unsqueeze(2)-D.unsqueeze(1)) ** 2).sum(dim=-1)
        D = distances.sqrt()
    
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
    
        # STGC_blocks
        # 1~3 STCC_blocks(distance_matrix)
        x = self.stgc1(x, self.A0_base, self.A0_distance, D)
        x = self.stgc2(x, self.A0_base, self.A0_distance, D)
        x = self.stgc3(x, self.A0_base, self.A0_distance, D)
        x = self.stgc4(x, self.A0_base)
        x = self.stgc5(x, self.A0_base)
        x = self.stgc6(x, self.A0_base)
    
        # Prediction
         x_shape = [int(s) for s in x.shape[2:]]
         x = F.avg_pool2d(x, x_shape)
         x = x.view(N, -1, 1, 1)
         x = self.fc(x)
         x = x.view(x.size(0), -1)

        return x
