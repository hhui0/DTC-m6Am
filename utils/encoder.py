import torch
from torch.nn import functional as F
class iFunction:

    def __init__(self) -> None:
        """"""

    @staticmethod
    def to_number(seq):
        base_dict = {
            'A': 0, 'C': 1, 'G': 2, 'T': 3
        }
        return torch.LongTensor([base_dict[c] for c in seq]).float().unsqueeze(-1)

    @staticmethod
    def to_eiip(seq):
        base_dict = {
            'A': 0.1260, 'T': 0.1335, 'C': 0.1340, 'G': 0.0806
        }
        return torch.tensor([base_dict[c] for c in seq]).float().unsqueeze(-1)

    @staticmethod
    def to_ncp(seq):
        base_dict = {
            'A': [1, 1, 1], 'T': [0, 0, 1], 'C': [0, 1, 0], 'G': [1, 0, 0]
        }
        return torch.tensor([base_dict[c] for c in seq]).float()

    @staticmethod
    def to_nd(seq):
        count_dict = {
            'A': 0, 'T': 0, 'C': 0, 'G': 0
        }
        res = []
        for i, (c,) in enumerate(seq):
            count_dict[c] += 1
            res.append(count_dict[c] / (i + 1))
        return torch.tensor(res).float().unsqueeze(-1)


    @staticmethod
    def fe_one_ncp(seq):
        x1 = iFunction.to_one_hot(seq)
        x3 = iFunction.to_ncp(seq)
        x = torch.cat([x1, x3], dim=-1)
        x.numpy()
        return x

    @staticmethod
    def fe_one_eiip(seq):
        x1 = iFunction.to_one_hot(seq)
        x3 = iFunction.to_eiip(seq)
        x = torch.cat([x1, x3], dim=-1)
        x = x.numpy()
        return x

    @staticmethod
    def fe_one_nd(seq):
        x1 = iFunction.to_one_hot(seq)
        x3 = iFunction.to_nd(seq)
        x = torch.cat([x1, x3], dim=-1)
        x = x.numpy()
        return x

    @staticmethod
    def fe_ncp_nd(seq):
        x2 = iFunction.to_ncp(seq)
        x3 = iFunction.to_nd(seq)
        x = torch.cat([x2, x3], dim=-1)
        x = x.numpy()
        return x
    @staticmethod
    def to_one_hot(seq):
        base_dict = {
            'A': 0, 'C': 3, 'G': 2, 'T': 1
        }
        return F.one_hot(torch.tensor([base_dict.get(c, 4) for c in seq]), 4).float()

    @staticmethod
    def fe_one_ncp_nd(seq):
        x1 = iFunction.to_one_hot(seq)
        x2 = iFunction.to_ncp(seq)
        x3 = iFunction.to_nd(seq)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = x.numpy()
        return x







