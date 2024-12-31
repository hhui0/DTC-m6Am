import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DenseLayer(in_channels, growth_rate))
            in_channels += growth_rate  # Update in_channels for the next layer

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat((x, new_features), dim=1)  # Concatenate along channel dimension
        return x

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv(out)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, num_layers, input_channels=5, output_channels=12,k_size=3):
        super(DenseNet, self).__init__()
        self.num_blocks = num_blocks

        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, growth_rate * 2, kernel_size=k_size, padding=1)

        # Dense blocks and transition layers
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        in_channels = growth_rate * 2

        for _ in range(num_blocks):
            block = DenseBlock(in_channels, growth_rate, num_layers)
            self.blocks.append(block)
            in_channels += growth_rate * num_layers  # Update for the next block
            if _ < num_blocks - 1:  # No transition after the last block
                transition = Transition(in_channels, in_channels // 2)
                self.transitions.append(transition)
                in_channels //= 2

        self.final_bn = nn.BatchNorm1d(in_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv1d(in_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
            if i < self.num_blocks - 1:
                x = self.transitions[i](x)

        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.final_conv(x)
        return x




# 测试代码
if __name__ == '__main__':
    input_tensor = torch.randn(64, 4, 41)  # (batch_size, channels, sequence_length)
    model = DenseNet(num_blocks=3, growth_rate=30, num_layers=4, input_channels=4, output_channels=150)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # 输出形状
