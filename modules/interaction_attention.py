class InteractionAttention(nn.Module):
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
        N, C, T, V = x.size()
        
        x_left = x[:, :, :, :21]
        QKV_left = self.qkv_left(x_left)
        Q_left, K_left, V_left = torch.chunk(QKV_left, 3, dim=1)

        x_right = x[:, :, :, 21:]
        QKV_right = self.qkv_right(x_right)
        Q_right, K_right, V_right = torch.chunk(QKV_right, 3, dim=1)
        
        Q_left = Q_left.view(N, self.num_heads, self.channels_per_head, T, 21)
        K_left = K_left.view(N, self.num_heads, self.channels_per_head, T, 21)
        V_left = V_left.view(N, self.num_heads, self.channels_per_head, T, 21)

        Q_right = Q_right.view(N, self.num_heads, self.channels_per_head, T, 21)
        K_right = K_right.view(N, self.num_heads, self.channels_per_head, T, 21)
        V_right = V_right.view(N, self.num_heads, self.channels_per_head, T, 21)
        
        
        d_k_left = Q_right.size(2)
        attn_right_to_left = torch.matmul(Q_right.view(N, self.num_heads, T * 21, -1), K_left.view(N, self.num_heads, T * 21, -1).transpose(-2, -1)) / (d_k_left ** 0.5)
        attn_right_to_left = F.softmax(attn_right_to_left, dim=-1)
        out_right_to_left = torch.matmul(attn_right_to_left, V_left.view(N, self.num_heads, T * 21, -1))
        out_right_to_left = out_right_to_left.view(N, self.num_heads, T, 21, -1).permute(0, 1, 4, 2, 3).contiguous().view(N, -1, T, 21)

        d_k_right = Q_left.size(2)
        attn_left_to_right = torch.matmul(Q_left.view(N, self.num_heads, T * 21, -1), K_right.view(N, self.num_heads, T * 21, -1).transpose(-2, -1)) / (d_k_right ** 0.5)
        attn_left_to_right = F.softmax(attn_left_to_right, dim=-1)
        out_left_to_right = torch.matmul(attn_left_to_right, V_right.view(N, self.num_heads, T * 21, -1))
        out_left_to_right = out_left_to_right.view(N, self.num_heads, T, 21, -1).permute(0, 1, 4, 2, 3).contiguous().view(N, -1, T, 21)
        
        out = torch.cat([out_left_to_right, out_right_to_left], dim=3) 

        out = out.view(N, C, T, V)

        residual = out
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.dropout(out)
        out = out + self.residual(residual + x)
        
        return out
