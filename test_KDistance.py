import numpy as np
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from CLIP.clip_feature_dataset import clip_feature
from CLIP.CLIP_MCM import CLIP_MCM
import faiss
import faiss.contrib.torch_utils
from CLIP.CLIP_ft import CLIP_ft


# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, cal_metric, print_measures, print_measures_with_std
    import utils.score_calculation as lib

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=256)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='/nobackup-slow/taoleitian/model/vos/ImageNet-100/MCM/vis/12/', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='energy', type=str, help='score options: MSP|energy')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--save', type=bool, default=False, help='save the results in the txt')
parser.add_argument('--model_name', default='res', type=str)
parser.add_argument('--num_layers', type=int, default=10)

args = parser.parse_args()
print(args)
# torch.manual_seed(1)
# np.random.seed(1)
num_layers = args.num_layers
# mean and standard deviation of channels of CIFAR-10 images
train_transform = trn.Compose([
    #trn.Resize(size=224, interpolation=trn.InterpolationMode.BICUBIC),
    trn.RandomResizedCrop(size=(224, 224), scale=(0.5, 1), interpolation=trn.InterpolationMode.BICUBIC),
    trn.RandomHorizontalFlip(p=0.5),
    trn.ToTensor(),
    trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
test_transform = trn.Compose([
    trn.Resize(size=(224, 224), interpolation=trn.InterpolationMode.BICUBIC),
    trn.CenterCrop(size=(224, 224)),
    trn.ToTensor(),
    trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

if 'cifar10_' in args.method_name:
    test_data = dset.CIFAR10('/nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform, download=True)
    num_classes = 10

elif 'ImageNet-100_' in args.method_name:
    load_path = '/nobackup-slow/taoleitian/CLIP_visual_feature/ImageNet-100/'+ str(num_layers)
    test_data = clip_feature(path=load_path+'/val/')
    train_data = clip_feature(path=load_path+'/train/')
    num_classes = 100
elif 'ImageNet-10_' in args.method_name:
    train_data = dset.ImageFolder('/nobackup-slow/dataset/ImageNet10/train', transform=test_transform)
    test_data = dset.ImageFolder('/nobackup-slow/dataset/ImageNet10/val', transform=test_transform)
    num_classes = 10
else:

    test_data = dset.CIFAR100('/nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform, download=True)
    num_classes = 100
# Create model
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=True,
                                          num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)
net = CLIP_ft(num_classes=num_classes, layers=args.num_layers)
#net = clipnet_ft(num_classes=num_classes, layers=args.num_layers)

start_epoch = 0

# Restore model

if args.load != '':
    for i in range(1000 - 1, -1, -1):
        if 'pretrained' in args.method_name:
            subdir = 'pretrained'
        elif 'oe_tune' in args.method_name:
            subdir = 'oe_tune'
        elif 'energy_ft' in args.method_name:
            subdir = 'energy_ft'
        elif 'baseline' in args.method_name:
            subdir = 'baseline'
        else:
            subdir = 'oe_scratch'

        model_name = os.path.join(args.load, args.method_name + '_epoch_' + str(i) + '.pt')
        # model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '.pt')
        if os.path.isfile(model_name):
            #net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_name).items()})
            #net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume " + model_name

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) * 2
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

correct = 0
feature_list = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > len(train_loader):
            break
        data, target = data.cuda(), target.cuda()

        # forward
        output, feature, _ = net(data)
        loss = F.cross_entropy(output, target)
        feature_norm = feature / feature.norm(dim=-1, keepdim=True)
        feature_list.append(feature_norm)
        # accuracy
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()

feature_list = torch.cat(feature_list, dim=0)
correct_rate = correct / len(feature_list)
print(correct_rate)
res = faiss.StandardGpuResources()

KNN_index = faiss.GpuIndexFlatL2(res, 512)

KNN_index.add(feature_list)

auroc_list, aupr_list, fpr_list = [], [], []

def get_scores(data_loader):
    ood_score = []
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(data_loader):

            data = data.cuda()

            output, feature, _ = net(data)
            feature_norm = feature / feature.norm(dim=-1, keepdim=True)
            D, _ = KNN_index.search(feature_norm, 100)
            D = -D[:, -1]
            ood_score.append(D)
    ood_score = torch.cat(ood_score, dim=0)
    return to_np(ood_score)

in_score = get_scores(test_loader)
def get_and_print_results(ood_loader):
    aurocs, auprs, fprs = [], [], []

    out_score = get_scores(ood_loader)
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0])
    auprs.append(measures[1])
    fprs.append(measures[2])
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    print_measures(auroc, aupr, fpr, args.method_name)
# /////////////// iNaturalist ///////////////
ood_path = '/nobackup-slow/taoleitian/CLIP_visual_feature/iNaturalist/'+str(num_layers)+'/'
ood_data = clip_feature(ood_path)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=4, pin_memory=True)
print('\n\niNaturalist Detection')
get_and_print_results(ood_loader)


# /////////////// Places /////////////// # cropped and no sampling of the test set
ood_path = '/nobackup-slow/taoleitian/CLIP_visual_feature/Places/'+str(num_layers)+'/'
ood_data = clip_feature(ood_path)

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nPlaces Detection')
get_and_print_results(ood_loader)

# /////////////// SUN ///////////////
ood_path = '/nobackup-slow/taoleitian/CLIP_visual_feature/SUN/'+str(num_layers)+'/'
ood_data = clip_feature(ood_path)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nSUN Detection')
get_and_print_results(ood_loader)

# /////////////// Textures ///////////////
ood_path = '/nobackup-slow/taoleitian/CLIP_visual_feature/Textures/'+str(num_layers)+'/'
ood_data = clip_feature(ood_path)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\nTextures Detection')
get_and_print_results(ood_loader)


print('\n\nMean Test Results!!!!!')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)