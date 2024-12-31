from models.TCN import TemporalConvNet
import torch.nn as nn
from typing import Dict, Any
from models.MLPs import MLP

from models.Densenet import DenseNet


#测试模块导入
from models.CBAM import CBAMBlock

import json

# 从json文件中读取参数
with open('config.json', 'r') as f:
    config = json.load(f)



class classifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_channels,
        dropout,
        kernel_size,
        L,
        MHSA_dim,
        r=10,
        num_linears=[64],
        dropout_linears=[0.2],
        mhsa_drop=0.1,
        seqlen=41,
        n_heads=3,
        reduction=11
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            #卷积层
            # nn.Conv1d(in_channels, num_channels[0], kernel_size=3, stride=1,padding=1),
            # MBConvBlock1D(ksize=3, input_filters=in_channels, output_filters=num_channels[0], image_size=41),
            # SKConv(in_channels, num_channels[0], 1, 2, 1, r, L, False, True, False),
            # CondConv(in_planes=in_channels, out_planes=num_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            # DynamicConv(in_planes=in_channels, out_planes=num_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            # DepthwiseSeparableConvolution1D(in_channels, num_channels[0]),
            DenseNet(num_blocks=1, growth_rate=30, num_layers=4, input_channels=in_channels, output_channels=num_channels[0]),

            # 长期依赖层
            TemporalConvNet(
                num_inputs=num_channels[0], num_channels=num_channels[1:], kernel_size=kernel_size, dropout=dropout
            ),

            #特征选择层（恒等映射）
            CBAMBlock(channel=num_channels[-1], reduction=reduction, kernel_size=3),
            # MultiHeadedSelfAttention(num_channels[-1], MHSA_dim * n_heads, mhsa_drop, n_heads),
            # DAModule(d_model=num_channels[0], seq_length=41),
            # ECAAttention(kernel_size=3, d_model=num_channels[0], seq_length=41),
            # SEAttention(channel=num_channels[0], reduction=reduction),


            #展开
            nn.Flatten(),
        )
        input_linear = seqlen * num_channels[-1]
        self.net.append(MLP(input_linear, num_linears, dropout_linears, acti="hardswish"))



    def forward(self, X):
        final_result=self.net(X.permute(0, 2, 1))
        return final_result

    def get_code(self, X):
        return self.net[0](X.permute(0, 2, 1))

    @staticmethod
    def get_model_params():
        model_params = dict(
            in_channels=config["in_channels"],
            MHSA_dim=11,
            n_heads=3,
            num_channels=[150, 100, 100, 100, 80,33],
            num_linears=[700, 220],
            kernel_size=3,
            mhsa_drop=0.1,
            L=78,
            r=6,
            seqlen=41,
            dropout=0.1,
            dropout_linears=[0.4, 0.2],
            reduction=30
        )
        return model_params

    @staticmethod
    def get_hparams() -> Dict[str, Any]:
        hparams = dict(batchsize=64, lr=config['lr'], patience=50, monitor=config["monitor"], name=config["model_name"], max_epochs=config['epoch'])
        return hparams

if __name__=='__main__':
    model_params=classifier.get_model_params()
    print(model_params)
    model=classifier(**model_params)

