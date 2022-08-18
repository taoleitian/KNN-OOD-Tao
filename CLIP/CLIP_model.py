import clip
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable

class clipnet(nn.Module):
    def __init__(self, num_classes=100, layers=8, type='ViT-B/16', device='cuda'):
        super(clipnet, self).__init__()
        model, preprocess = clip.load(type, device)
        self.imagenet_templates = [
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ]
        self.class_name = ['brambling bird', 'bull frog', 'thunder snake', 'Swiss mountain dog', 'Siamese cat', 'antelope', 'container ship', 'garbage truck', 'sports car', 'warplane']
        self.device = device

        self.fc = nn.Linear(512, num_classes)
        #self.image_encoder = model
        #for parm in self.image_encoder.parameters():
            #parm.requires_grad = False
        self.MLP = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.weight_energy = torch.nn.Linear(num_classes, 1).cuda()
        torch.nn.init.uniform_(self.weight_energy.weight)

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation

        value.exp().sum(dim, keepdim).log()
        """
        import math
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            energy_score = m + torch.log(torch.sum(
                F.relu(self.weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
            return self.MLP(energy_score.view(-1, 1))
            #return energy_score.view(-1, 1)
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)

    def forward(self, input, fc=False):
        if fc==True:
            output = self.fc(input)
            return output, self.log_sum_exp(output, 1)
        output = self.fc(input.float())
        return output, input, self.log_sum_exp(output, 1)
