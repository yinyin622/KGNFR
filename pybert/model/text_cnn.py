# file: text_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, hidden_size, num_filters, filter_sizes, num_labels, dropout=0.1):
        super(TextCNN, self).__init__()
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        # 多个一维卷积层，每个对应不同的 n-gram 大小
        # 输入通道数 = hidden_size (如 768)
        # 输出通道数 = num_filters (如 256)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=num_filters,
                kernel_size=fs
            ) for fs in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 分类用的全连接层
        # 总特征数 = num_filters * len(filter_sizes)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size] 例如 [B, 512, 768]
        
        # 转换为 CNN 输入格式：[B, hidden_size, seq_len]
        x = x.transpose(1, 2)  # -> [B, H, L]
        
        # 对每个卷积核进行卷积 + 激活 + 池化
        conv_outputs = []
        for conv in self.convs:
            # conv(x): [B, num_filters, seq_len - kernel_size + 1]
            cnn_out = conv(x)  # [B, C, L']
            
            # 激活函数
            cnn_out = F.relu(cnn_out)  # [B, C, L']
            
            # Max-over-time pooling: 在时间维度上取最大值 -> [B, C]
            pooled = F.max_pool1d(cnn_out, kernel_size=cnn_out.size(2)).squeeze(2)
            
            conv_outputs.append(pooled)  # 每个是 [B, C]

        # 将所有卷积结果拼接起来
        concat = torch.cat(conv_outputs, dim=1)  # [B, C * len(filter_sizes)]

        # Dropout
        concat = self.dropout(concat)
        
        # 全连接输出 logits
        logits = self.classifier(concat)  # [B, num_labels]
        
        return logits