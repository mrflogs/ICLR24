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
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from torch.autograd import Variable
import numpy as np

from sklearn.covariance import LedoitWolf, OAS, GraphicalLassoCV, GraphicalLasso

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

def run(cfg, train_loader_cache, clip_weights, clip_model):  
    
    # Parameter Estimation.
    with torch.no_grad():      
        # Ours
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
        
        # normal distribution
        mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(clip_weights.shape[1])])

        # KS Estimator.  
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
        
        print("best_val_alpha: %s \t best_val_acc: %s" % (best_alpha, best_val_acc))
        alpha = best_alpha
        test_logits = 100. * test_features.float() @ clip_weights.float() + alpha * (test_features.float() @ W + b)
        notune_acc = cls_acc(test_logits, test_labels)    
        print("Nonetune acc:", notune_acc)
    return notune_acc

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
        
    notune_accs = {"1": [], "2": [], "3": []}
    
    for seed in [1, 2, 3]:
        cfg["seed"] = seed
        random.seed(seed)
        torch.manual_seed(seed) 
        
        for shots in [1, 2, 4, 8, 16]:
            cfg["shots"] = shots
        
            print("Preparing dataset.")
            global train_loader_F
            global test_features, test_labels
            global val_features, val_labels
            if cfg['dataset'] != "imagenet":
                dataset = build_dataset(cfg, cfg['dataset'], cfg['root_path'], cfg['shots']) 

                train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
                train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True) 

                test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)
                val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)

                test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
                val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

            else:
                dataset = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
                # train
                train_loader_F = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=True)
                train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)

                test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)

                test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)   
                val_features, val_labels = test_features, test_labels    

            clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model.float())   

            notune_acc = run(cfg, train_loader_cache, clip_weights, clip_model)
            notune_accs[str(cfg["seed"])].append(notune_acc)
    print("Evaluate on dataset:", cfg['dataset'])
    print("Evaluate on seed [1, 2, 3]")
    print("Evaluate on shots [1, 2, 4, 8, 16]")
    print("No tuning:")
    notune = []
    for seed in ["1", "2", "3"]:
        print("seed %s" % seed, notune_accs[str(seed)])
        notune.append(notune_accs[seed])
    notune = torch.tensor(notune)
    print("Average: ", notune.mean(dim=0))
    
if __name__ == '__main__':
    main()