import timm
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