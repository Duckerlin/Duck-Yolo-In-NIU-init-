import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)  # 输入通道3，输出通道64

    def forward(self, x):
        return self.conv1(x)

model = SimpleModel()
input_tensor = torch.randn(1, 3, 640, 640)  # 测试输入
output_tensor = model(input_tensor)  # 确保没有错误
print(output_tensor.shape)
