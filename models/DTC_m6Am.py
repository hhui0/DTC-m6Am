from models.TCN import TemporalConvNet
import torch.nn as nn
from typing import Dict, Any
from models.MLPs import MLP
from models.Densenet import DenseNet
from models.CBAM import CBAMBlock
import json

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
            DenseNet(num_blocks=1, growth_rate=30, num_layers=4, input_channels=in_channels, output_channels=num_channels[0]),
            TemporalConvNet(
                num_inputs=num_channels[0], num_channels=num_channels[1:], kernel_size=kernel_size, dropout=dropout
            ),
            CBAMBlock(channel=num_channels[-1], reduction=reduction, kernel_size=3),
            # MultiHeadedSelfAttention(num_channels[-1], MHSA_dim * n_heads, mhsa_drop, n_heads),
            # SEAttention(channel=num_channels[0], reduction=reduction),
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
        hparams = dict(batchsize=config["batch_size"], lr=config['lr'], patience=50, monitor=config["monitor"], name=config["model_name"], max_epochs=config['epoch'])
        return hparams

if __name__=='__main__':
    model_params=classifier.get_model_params()
    print(model_params)
    model=classifier(**model_params)

