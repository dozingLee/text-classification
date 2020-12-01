# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    # 初始化模型参数
    def __init__(self, args):
        super(TextCNN, self).__init__()

        # 模型全部的参数
        self.args = args

        # 标签词汇表的长度（由数据集确定）
        label_num = args.label_num

        # 卷积核的个数（由使用者设定）
        filter_num = args.filter_num

        # 卷积核大小数组（论文标准为[3,4,5]）
        filter_sizes = [int(fsz) for fsz in args.filter_sizes.split(',')]

        # 文本词汇表的长度（由数据集确定）
        vocab_size = args.vocab_size

        # 词向量的维度（由词向量表确定，根据具体使用的词向量的维度确定）
        embedding_dim = args.embedding_dim

        # 词向量矩阵
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 静态词向量（如果使用预训练，词向量则提前加载，当不需要微调时设置freeze为True）
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        # 神经网络模型
        self.conv = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, embedding_dim)) for fsz in filter_sizes])

        # dropout（随机丢弃，论文标准为0.5）
        self.dropout = nn.Dropout(args.dropout)

        # 确定最后的输出向量维度：len(filter_sizes) * filter_num
        self.linear = nn.Linear(len(filter_sizes) * filter_num, label_num)

    # 前馈网络
    def forward(self, x):

        # 输入x的维度为 (batch_size, max_len)
        #   max_len可以通过torchtext设置或自动获取为训练样本的最大长度

        # 经过embedding,x的维度为 (batch_size, max_len, embedding_dim)
        x = self.embedding(x)

        # 经过view函数x的维度变为 (batch_size, input_channel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.args.embedding_dim)

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_channel, w=max_len, h=1)
        x = [F.relu(conv(x)) for conv in self.conv]

        # 经过最大池化层,维度变为(batch_size, out_channel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]

        # 将不同卷积核运算结果维度 (batch, out_channel, w=1, h=1) 展平为 (batch, out_channel * w * h)
        x = [x_item.view(x_item.size(0), -1) for x_item in x]

        # 将不同卷积核提取的特征组合起来,维度变为 (batch, sum:out_channel * w * h)
        x = torch.cat(x, 1)

        # dropout层
        x = self.dropout(x)

        # 全连接层
        logistic = self.linear(x)
        return logistic

        # forward的另外一种写法
        # x = self.embedding(x)
        # x = x.unsqueeze(1)
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]
        # x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        # x = torch.cat(x, 1)
        # x = self.dropout(x)
        # logistic = self.fc(x)
        # return logistic
