import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import BCELoss

# 每个损失函数都要在这里注册！
__call__ = ['CrossEntropy', 'MultiLabelCrossEntropy', 'BCEWithLoss', 
            'BCEWithLogLoss', 'WeightedBCEWithLogLoss', 'FocalLoss']

# 交叉熵 多用于单标签分类
class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss


# 交叉熵的多标签分类版本
class MultiLabelCrossEntropy(object):
    def __init__(self):
        pass
    def __call__(self, output, target):
        loss = CrossEntropyLoss(reduction='none')(output,target)
        return loss
    

# BCE 原始的，多用于多标签分类
class BCEWithLoss(object):
    def __init__(self):
        self.loss_fn = BCELoss()

    def __call__(self,output,target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input = output,target = target)
        return loss


# BCE 的改进版 BCEWithLogLoss（Python官方）
class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self,output,target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input = output,target = target)
        return loss

# BCE 的改进版++ （王WH）
class WeightedBCEWithLogLoss(object):
    def __init__(self, pos_weights=None):
        """
        Weighted BCE loss for handling class imbalance
        Args:
            pos_weights: torch.Tensor of shape [num_classes] containing weights for positive examples
        """
        self.loss_fn = BCEWithLogitsLoss(pos_weight=pos_weights)

    def __call__(self, output, target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input=output, target=target)
        return loss


# 长尾分布数据集，单标签分类 FocalLoss 的改进版（邓JQ）
class FocalLoss(object):
    def __init__(self, gamma=1, alpha=None):
        """
        Focal Loss for addressing extreme class imbalance
        Args:
            gamma: focusing parameter that adjusts rate at which easy examples are down-weighted
            alpha: weighting factor for rare classes (can be a tensor or scalar)
        """
        self.gamma = gamma
        self.alpha = alpha
        
    def __call__(self, output, target):
        output = output.float()
        target = target.float()
        
        # Use sigmoid since we're in multi-label setting
        probs = torch.sigmoid(output)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        
        # Apply focal weighting
        pt = torch.where(target == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            # 将alpha转换为tensor
            alpha_tensor = torch.tensor(self.alpha, device=target.device)
            # 只对正样本应用alpha权重，负样本保持为1
            alpha_weight = torch.where(target == 1, alpha_tensor, torch.ones_like(alpha_tensor))
            # alpha_weight = torch.where(target == 1, alpha_tensor, 1-alpha_tensor)
            focal_weight = alpha_weight * focal_weight
            
        # 计算损失并返回平均值
        return (focal_weight * bce_loss).mean()






