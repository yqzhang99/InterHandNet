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


