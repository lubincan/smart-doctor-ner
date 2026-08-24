import torch
# 假设 input_size 为 5
input_size = 5
# 创建一个表示单个时间步的特征的张量
single_time_step = torch.randn(input_size)  # 形状: [5]
print('single_time_step=',single_time_step)
# 增加批次维度
input_tensor = single_time_step.unsqueeze(0)  # 形状: [1, 5]
print('input_tensor=',input_tensor)
# 如果 RNN 需要 (batch_size, sequence_length, input_size)
input_tensor = input_tensor.unsqueeze(0)  # 形状: [1, 1, 5]
print('input_tensor2=',input_tensor)
print(input_tensor.shape)  # 输出: torch.Size([1, 1, 5])


import matplotlib
matplotlib.use('TkAgg')  # 或者 'Agg'
import matplotlib.pyplot as plt

# 示例数据
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.plot(x, y)
plt.title('Sample Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图形（如果使用 TkAgg）
plt.show()

# 保存图形（如果使用 Agg）
# plt.savefig('output.png')


