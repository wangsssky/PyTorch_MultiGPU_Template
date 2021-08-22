import torch
import torch.nn as nn
import torchvision.models as model_pool


# model samples
class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.model = model_pool.resnet18(pretrained=False)

    def forward(self, x):
        output = self.model(x)
        return output


class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        self.model = model_pool.resnet50(pretrained=False)

    def forward(self, x):
        output = self.model(x)
        return output


NAME_TO_FUNC = {
    'resnet18': resnet18,
    'resnet50': resnet50,
}


def build_model(opt):
    # specify modal key
    model_key = opt.arch

    model = NAME_TO_FUNC[model_key]()
    if opt.use_ema:
        model_ema = NAME_TO_FUNC[model_key]()
    else:
        model_ema = None

    return model, model_ema
