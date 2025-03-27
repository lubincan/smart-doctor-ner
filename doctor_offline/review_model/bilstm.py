import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tag_to_id, input_feature_size, hidden_size,
                 batch_size, sentence_length,num_layer=1,batch_first=True):
        '''
        vocab_size:   所有字句包含字符大小
        tag_to_id:    标签与id对照
        input_feature_size:  字嵌入维度（即LSTM输入层维度input_size）
        hidden_size:    隐藏层的维度
        batch_size:    训练批大小
        sentence_length:    句子长度
        num_layer:  堆叠LSTM层数量
        batch_first：是否将batch_size放置到矩阵的第一维度
        '''

        # 继承函数的初始化
        super(BiLSTM, self).__init__()

        self.vocab_size = vocab_size

        self.tag_to_id = tag_to_id

        self.tag_size = len(tag_to_id)

        self.embedding_size = input_feature_size

        self.hidden_size = hidden_size//2

        self.batch_size = batch_size

        self.sentence_length = sentence_length

        self.num_layer = num_layer

        self.batch_first = batch_first

        # 构建词嵌入层, 两个参数分别单词总数量, 词嵌入维度
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)

        # 构建双向LSTM层, 输入参数包括词嵌入维度, 隐藏层大小, LSTM层数, 是否双向标志
        self.bilstm = nn.LSTM(input_size=input_feature_size,
                            hidden_size=self.hidden_size,num_layers=self.num_layer,bidirectional=True,batch_first=self.batch_first)

        self.liner = nn.Linear(hidden_size,self.tag_size)


    # 编写正式的forward()函数, 注意应用场景是在预测的时候, 模型训练的时候并没有用到forward()函数
    def forward(self, sentence_sequence):
        #
        h0 = torch.randn(self.num_layer * 2,self.batch_size,self.hidden_size)
        c0 = torch.randn(self.num_layer * 2, self.batch_size,self.hidden_size)
        input_features = self.embedding(sentence_sequence)
        output,(hn,cn) = self.bilstm(input_features,(h0,c0))
        sequence_features = self.liner(output)
        return  sequence_features

# 参数1:码表与id
char_to_id ={"双":0,"肺":1,"见":2,"多":3,"发":4,"斑":5,"片":6,"状":7,"稍":8,"高":9,"密":10,"度":11,"影":12,"。":13}

# 参数2:标签码表对照
tag_to_id ={"0":0,"B-dis":1,"I-dis":2,"B-sym":3,"I-sym": 4}
#参数3:字向最维度
EMBEDDING_DIM=200

# 参数4.隐层维度
HIDDEN_DIM=100
#参数5:批次大小
BATCH_SIZE=8
#参数6:句子长度
SENTENCE_LENGTH =20
# 参数7:堆叠 L5TM 层数
NUM_LAYERS =1
# 设置最大语句限制长度

#调用
model = BiLSTM(vocab_size=len(char_to_id),
               tag_to_id=tag_to_id,
               input_feature_size=EMBEDDING_DIM,
               hidden_size=HIDDEN_DIM,
               batch_size=BATCH_SIZE,
               sentence_length=SENTENCE_LENGTH,
               num_layer=NUM_LAYERS)

print(model)

# 函数sentence_map()完成中文文本信息的数字编码, 将中文语句变成数字化张量,采取手动填充的方式
def sentence_map(sentence_list, char_to_id, max_length):
    # 首先对一个批次的所有语句按照句子的长短进行排序, 这个操作并非必须，排序是为了在填充时提升效率，节约内存
    sentence_list.sort(key=lambda x: len(x), reverse=True)

    # 定义一个最终存储结果特征张量的空列表
    sentence_map_list = []
    # 循环遍历一个批次内所有的语句
    for sentence in sentence_list:
        # 采用列表生成式来完成中文字符到id值的映射
        sentence_id_list = [char_to_id[c] for c in sentence]
        #print(sentence_id_list)
        # 长度不够max_length的部分用0填充，手动填充
        padding_list = [0] * (max_length - len(sentence))
        # 将每一个语句扩充为相同长度的张量
        sentence_id_list.extend(padding_list)
        # 追加进最终存储结果的列表中
        sentence_map_list.append(sentence_id_list)

    # 返回一个标量类型的张量
    return torch.tensor(sentence_map_list, dtype=torch.long)

sentence_list = [
    "确诊弥漫大b细胞淋巴瘤1年",
    "反复咳嗽、咳痰40年,再发伴气促5天。",
    "生长发育迟缓9年。",
    "右侧小细胞肺癌第三次化疗入院",
    "反复气促、心悸10年,加重伴胸痛3天。",
    "反复胸闷、心悸、气促2多月,加重3天",
    "咳嗽、胸闷1月余, 加重1周",
    "右上肢无力3年, 加重伴肌肉萎缩半年"
]

char_to_id = {"<PAD>": 0}
SENTENCE_LENGTH = 20
if __name__ == '__main__':
    for sentence in sentence_list:
        for c in sentence:
            # 如果当前字符不在映射字典中, 追加进字典
            if c not in char_to_id:
                char_to_id[c] = len(char_to_id)

    # 首先利用char_to_id完成中文文本的数字化编码
    sentence_sequence = sentence_map(sentence_list, char_to_id, SENTENCE_LENGTH)
    print("sentence_sequence:\n", sentence_sequence)
    model = BiLSTM(vocab_size=len(char_to_id), tag_to_id=tag_to_id, input_feature_size=EMBEDDING_DIM,
                       hidden_size=HIDDEN_DIM, batch_size=BATCH_SIZE, sentence_length=SENTENCE_LENGTH,
                       num_layer=NUM_LAYERS)
    sentence_features = model(sentence_sequence)
    print("sentence_features:\n", sentence_features)



