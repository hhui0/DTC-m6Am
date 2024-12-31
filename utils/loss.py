import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2.0,reduce='sum'):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self,classifications,targets):

        alpha = self.alpha
        gamma = self.gamma
        classifications = classifications.view(-1)
        targets = targets.view(-1)

        ce_loss = F.binary_cross_entropy_with_logits(classifications, targets.float(), reduction="none")
        #focal loss
        p = torch.sigmoid(classifications)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduce=='sum':
            loss = loss.sum()
        elif self.reduce=='mean':
            loss = loss.mean()
        else:
            raise ValueError('reduce type is wrong!')
        return loss