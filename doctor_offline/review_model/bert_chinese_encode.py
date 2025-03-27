import torch
import torch.nn as nn

# 导入bert的模型
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-chinese')

# 导入字符映射器
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-chinese')

def get_bert_encode_for_single(text):
    """
    功能: 使用bert-chinese预训练模型对中文文本进行编码
    text: 要进行编码的中文文本
    return : 编码后的张量
    """

    # 首先使用字符映射器对每个汉子进行映射
    # bert中的tokenizer映射后会加入开始和结束的标记, 101, 102, 这两个标记对我们不需要，采用切片的方式去除
    indexed_tokens = tokenizer.encode(text)[1:-1]
    #print(indexed_tokens)
    # 封装成tensor张量
    tokens_tensor = torch.tensor([indexed_tokens])
    #print(tokens_tensor)

    # 预测部分需要使得模型不自动求导
    with torch.no_grad():
        '''行代码假设模型返回的是两个部分，其中第一个部分是编码层（encoded_layers），而第二个部分（用 _ 表示）是我们不需要的输出（可能是池化层的输出）如果您只对模型的编码层输出感兴趣（通常是 last_hidden_state），这种写法是合理的。需要注意的是，这种写法依赖于模型的输出结构，如果模型返回的不是两个部分，可能会导致错误'''
        #encoded_layers, _ = model(**tokens_tensor)
        '''这行代码将模型的所有输出存储在 outputs 变量中。模型的输出通常是一个对象，包含多个属性（例如 last_hidden_state 和 pooler_output）这种写法更灵活，因为您可以使用 outputs 对象中的所有信息，而不仅仅是第一个输出部分。您可以访问最后的隐藏状态和池化输出'''
        outputs = model(tokens_tensor)

    # print(encoded_layers.shape)

    # 模型的输出都是三维张量,第一维是1,使用[0]来进行降维,只提取我们需要的后两个维度的张量
    encoded_layers = outputs.last_hidden_state  # 这是一个张量
    encoded_layers = encoded_layers[0]
    return encoded_layers


if __name__ == '__main__':
    text = "你好,周杰伦"
    outputs = get_bert_encode_for_single(text)
    #print(text)
    print(outputs)
    #print(outputs.shape)

