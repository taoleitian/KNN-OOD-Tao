import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class clipnet_ft(nn.Module):
    def __init__(self, num_classes=100, layers=8,  type='ViT-B/16', device='cuda'):
        super(clipnet_ft, self).__init__()
        model, preprocess = clip.load(type, device)
        self.layers = layers
        self.device = device

        self.fc = nn.Linear(512, num_classes).to(device)
        #self.model = model.visual

        self.ln_post = model.visual.ln_post
        self.transformer_resblocks_one = model.visual.transformer.resblocks[10]
        self.transformer_resblocks_two = model.visual.transformer.resblocks[11]
        self.proj = model.visual.proj

        self.weight_energy = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.MLP = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        #torch.nn.init.uniform_(self.weight_energy.weight)

    def compute_sigmoid(self, value):
        zero = self.sigmoid(self.MLP(value))
        one = torch.ones_like(zero) - zero
        return torch.cat([zero, one], dim=1)

    def uncertainty(self, value):
        energy_score = torch.logsumexp(value[:, :-1] / 1.0, 1)
        return self.MLP(energy_score.view(-1, 1))

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation

        value.exp().sum(dim, keepdim).log()
        """
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

    def visual_forward_one(self, x):
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2) # NLD -> LND
        for i in range(self.layers):
            x = self.model.transformer.resblocks[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x

    def visual_forward_two(self, x):
        x = x.permute(1, 0, 2) # NLD -> LND
        for i in range(self.layers, 12):
            x = self.model.transformer.resblocks[i](x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.model.ln_post(x[:, 0, :])

        if self.model.proj is not None:
            x = x @ self.model.proj
        return x

    def ft_blocks(self, x):
        x = x.permute(1, 0, 2) # NLD -> LND

        x = self.transformer_resblocks_one(x)
        x = self.transformer_resblocks_two(x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x

    def forward(self, input, fc=False):
        if fc==True:
            output = self.fc(input)
            score = self.weight_energy(input)
            return output, score

        feature = self.ft_blocks(input).float()
        output = self.fc(feature)
        score = self.weight_energy(feature)
        return output, feature, score