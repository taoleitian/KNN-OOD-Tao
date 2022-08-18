from CLIP_model import clipnet
from clip_feature_dataset import clip_feature
from CLIP_ft import clipnet_ft
import torch
ood_path = '/nobackup-slow/taoleitian/CLIP_visual_feature/ImageNet-100/10/train/'
# load_path = '/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/dataset/ImageNet-100/'
train_data = clip_feature(path=ood_path)
num_classes = 100
net = clipnet_ft(num_classes=num_classes, layers=10).cuda()
net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('/nobackup-slow/taoleitian/model/vos/ImageNet-100/epoch/10/0.01/600/6000/3/ImageNet-100_dense_baseline_dense_epoch_12.pt').items()})

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=1, shuffle=False,
    num_workers=8, pin_memory=True)
data_list = []
net.eval()
for idx, data in enumerate(train_loader):
    target = data[1]
    input = data[0].cuda()
    if target==0:
        print(idx)
        _, feature, _ = net(input)
        data_list.append(feature.detach().cpu())

data_save = torch.cat(data_list, dim=0)
torch.save(data_save, 'OOD_data.pt')
