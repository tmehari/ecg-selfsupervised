import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .xresnet1d import xresnet1d50, xresnet1d101


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, widen=1.0, hidden=False):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False), 
                            "xresnet1d50": xresnet1d50(widen=widen),
                            "xresnet1d101": xresnet1d101(widen=widen)}

        resnet = self._get_basemodel(base_model)
        self.base_model = base_model
    
        list_of_modules = list(resnet.children())
        if "xresnet" in base_model:
            self.features = nn.Sequential(*list_of_modules[:-1], list_of_modules[-1][0])
            num_ftrs = resnet[-1][-1].in_features
            resnet[0][0] = nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)
        else:
            self.features = nn.Sequential(*list_of_modules[:-1])
            num_ftrs = resnet.fc.in_features

        # projection MLP
        if hidden:
            self.l1 = nn.Linear(num_ftrs, num_ftrs)
            self.l2 = nn.Linear(num_ftrs, out_dim)
        else:
            self.l1 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
