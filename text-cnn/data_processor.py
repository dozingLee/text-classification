import jieba
from torchtext import data
import re
from torchtext.vocab import Vectors


# 句子分词
def tokenizer(text):  # create a tokenizer function
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')  # ^\u4e00-\u9fa 匹配汉字
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


# 获取停用词
def get_stop_words():
    file_object = open('data/stopwords.txt', 'rb')  # 直接读数据
    stop_words = []
    for line in file_object.readlines():  # 按行读取
        # line = line[:-1]    # line的元素从0至位置-1（倒数第一个），相当于处于最后一个字符
        line = line.strip()  # 去掉首位换行符
        stop_words.append(line)
    return stop_words


# 加载数据
def load_data(args):

    # 加载停用词表
    stop_words = get_stop_words()

    # 定义数据字段
    '''
    如果需要设置文本的长度，则设置fix_length,否则torchtext自动将文本长度处理为最大样本长度
    text = data.Field(sequential=True, tokenize=tokenizer, fix_length=args.max_len, stop_words=stop_words)
    '''
    text = data.Field(sequential=True, lower=True, tokenize=tokenizer, stop_words=stop_words)
    label = data.Field(sequential=False)

    # 读取数据并根据定义的字段加载数据
    train, val = data.TabularDataset.splits(
        path='data/',
        skip_header=True,
        train='train.tsv',
        validation='validation.tsv',
        format='tsv',
        fields=[('index', None), ('label', label), ('text', text)],
    )

    # 加载静态词向量
    if args.static:
        text.build_vocab(train, val, vectors=Vectors(name="data/eco_article.vector"))  # 此处改为你自己的词向量
        args.embedding_dim = text.vocab.vectors.size()[-1]
        args.vectors = text.vocab.vectors  # 返回词汇表
    # 构建动态词向量，训练自动调整
    else:
        text.build_vocab(train, val)

    # 此处与后面训练模型时标签-1相对应，并无太大意义
    # 标签build后：对于无标签<unk>为0，有标签0和1在build之后分别为1和2
    label.build_vocab(train, val)

    # 构建batch大小的数据集
    '''
    torchtext.data.Iterators
        Source: https://pytorch.org/text/stable/data.html
    
        (1) 这里sort_key根据文本的长度进行排序
            sort_key用于对示例进行排序的键，以便将具有相似长度的示例批量组合在一起并最小化填充
        (2) lambda可以理解为箭头函数，输入x返回x的text属性值的长度
        (3) 这里val验证集的batch_size则正好为验证集的长度，不用进行细分
        (4) 这里device用于指定创建变量的设备（默认为cpu），如果使用gpu需要指定设备字符串
    '''
    train_iter, val_iter = data.Iterator.splits(
        (train, val),
        sort_key=lambda x: len(x.text),  # text是文件中读取出的列主键
        batch_sizes=(args.batch_size, len(val)),  # 训练集设置batch_size,验证集整个集合用于测试
        device=-1
    )

    args.vocab_size = len(text.vocab)   # 文本词汇表的数量（单词转为不同数字的数量）
    args.label_num = len(label.vocab)   # 标签词汇表的数量

    return train_iter, val_iter
