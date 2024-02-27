import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets.imagenet import ImageNet
from datasets.imagenet_a import ImageNet_A
from datasets.imagenet_r import ImageNet_R
from datasets.imagenet_sketch import ImageNet_Sketch
from datasets.imagenet_v2 import ImageNet_V2
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from torch.autograd import Variable
import numpy as np

_tokenizer = _Tokenizer()
train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()
    return args  

def get_vecs(cfg, train_loader_cache, clip_model):
    vecs = []
    labels = []
    for i in range(cfg["augment_epoch"]):
        for image, target in tqdm(train_loader_cache):
            image, target = image.cuda(), target.cuda()
            image_features = clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            vecs.append(image_features)
            labels.append(target)
    vecs = torch.cat(vecs)
    labels = torch.cat(labels)
    return vecs, labels

def run(cfg, train_loader_cache, clip_weights, clip_model):  
    
    # Parameter Estimation.
    with torch.no_grad():      
        # Ours
        vecs, labels = get_vecs(cfg, train_loader_cache, clip_model)
        
        mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(clip_weights.shape[1])])
        center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(clip_weights.shape[1])])
        cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).cuda())
        
        ps = torch.ones(clip_weights.shape[1]).cuda() * 1. / clip_weights.shape[1]
        W = torch.einsum('nd, dc -> cn', mus, cov_inv)
        b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2
        
        # Evaluate
        # Grid search for hyper-parameter alpha
        best_val_acc = 0
        best_alpha = 0.1
        for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            val_logits = 100. * val_features.float() @ clip_weights.float() + alpha * (val_features.float() @ W + b)
        
            acc = cls_acc(val_logits, val_labels)
            if acc > best_val_acc:
                best_val_acc = acc
                best_alpha = alpha
        alpha = best_alpha 
        print("ImageNet Acc:", best_val_acc)
        
    return best_val_acc, vecs, alpha, W, b

def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    # Load cfg for conditional prompt.
    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    
    accs = {"1": [], "2": [], "3": []}
    for seed in [1, 2, 3]:
        cfg["seed"] = seed
        random.seed(seed)
        torch.manual_seed(seed)       
        
        print("Preparing dataset.")
        global train_loader_F
        global test_features, test_labels
        global val_features, val_labels
        
        # Source dataset
        cfg['dataset'] = 'imagenet'
        dataset = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
        # train
        train_loader_F = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=True)
        train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)

        val_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
        val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)    

        clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model.float())   

        acc, vecs, alpha, W, b = run(cfg, train_loader_cache, clip_weights, clip_model)
        accs[str(seed)].append(acc)
        
        # Target dataset
        target_datasets = ['imagenet-v2', 'imagenet-sketch', 'imagenet-a', 'imagenet-r']
        dataset_list = {
            'imagenet-v2': ImageNet_V2,
            'imagenet-sketch': ImageNet_Sketch,
            'imagenet-a': ImageNet_A,
            'imagenet-r': ImageNet_R
        }
        
        for target_dataset in target_datasets:
            cfg['dataset'] = target_dataset
            dataset = dataset_list[target_dataset](cfg, cfg['root_path'], cfg['shots'], preprocess) 
            test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
            test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
            label_mapping = dataset.label_mapping
            
            test_logits = 100. * test_features.float() @ clip_weights.float() + alpha * (test_features.float() @ W + b)
            test_logits = test_logits @ label_mapping
            target_acc = cls_acc(test_logits, test_labels)
            print("Target dataset: %s, acc: %s" % (target_dataset, target_acc))
            accs[str(seed)].append(target_acc)
            
    print("Evaluate on source & target dataset:", ['imagenet'] + target_datasets)
    print("Evaluate on seed [1, 2, 3]")
    print(accs)
    acc = []
    for seed in ["1", "2", "3"]:
        print("seed %s" % seed, accs[str(seed)])
        acc.append(accs[seed])
    acc = torch.tensor(acc)
    print("Average: ", acc.mean(dim=0))

    
if __name__ == '__main__':
    main()