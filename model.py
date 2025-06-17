import timm
import torch
import torch.nn.functional as F
from torch import nn

# Model Define (inf.py에서도 사용)
class CustomTimmModel(nn.Module):
    def __init__(self, model_name, num_classes_to_predict, pretrained=True):
        super(CustomTimmModel, self).__init__()
        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
            self.feature_dim = self.backbone.num_features
        except Exception as e:
            print(f"Error creating model {model_name} with timm. Error: {e}")
            raise
        self.head = nn.Linear(self.feature_dim, num_classes_to_predict)
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logit = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (one_hot * target_logit + (1.0 - one_hot) * cosine)
        return output

class FineGrainedModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.feature_dim = self.backbone.num_features
        self.bn = nn.BatchNorm1d(self.feature_dim)
        self.head = ArcMarginProduct(self.feature_dim, num_classes)

    def forward(self, x, labels):
        features = self.backbone(x)
        features = self.bn(features)
        output = self.head(features, labels)
        return output, features