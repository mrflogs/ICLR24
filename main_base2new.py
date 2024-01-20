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
from sklearn.covariance import LedoitWolf, OAS

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

def run(cfg, train_loader_cache, clip_weights, clip_weights_new, clip_model):  
    
    # Parameter Estimation.
    with torch.no_grad():     
        
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
        
        # Ours
        mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(clip_weights.shape[1])])
        train_cov = torch.cat([vecs[labels == i].T.cov().unsqueeze(0) for i in range(clip_weights.shape[1])])         
        cov_inv = torch.linalg.pinv(train_cov.mean(dim=0)) 
        
        ps = torch.ones(clip_weights.shape[1]).cuda() * 1. / clip_weights.shape[1]
        W = torch.einsum('nd, dc -> cn', mus, cov_inv)
        b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2
        
        
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
        # Evaluate
        # base
        test_logits = 100. * test_features.float() @ clip_weights.float() + alpha * (test_features.float() @ W + b)
        base_acc = cls_acc(test_logits, test_labels)
        
        # new
        # Synthesize new classifier (W, b)
        # topk covariance.
        k = 64
        mu = torch.cat([
            torch.cat([
                vecs[torch.topk(w @ vecs.T, k=k, largest=True)[1]], 
                w.unsqueeze(0).repeat(16, 1)
            ]).mean(dim=0, keepdim=True) 
            for w in clip_weights_new.T
        ])     
        
        def covariance(vecs, mu=0):
            N, D = vecs.shape
            cov = (vecs - mu).T @ (vecs - mu) / (N - 1)
            return cov

        # KS estimator
        cov_inv = torch.cat([
            512 * 
            torch.linalg.pinv(
                (k - 1) * covariance(vecs[torch.topk(w @ vecs.T, k=k, largest=True)[1]], mu=w.unsqueeze(0)) + 
                covariance(vecs[torch.topk(w @ vecs.T, k=k, largest=True)[1]], mu=w.unsqueeze(0)).trace() * torch.eye(512).cuda()
            ).unsqueeze(0)
            for i, w in enumerate(clip_weights_new.T)
        ]).mean(dim=0)   
        
        ps = torch.ones(clip_weights_new.shape[1]).cuda() * 1. / clip_weights_new.shape[1]
        W_new = torch.einsum('nd, dc -> cn', mu, cov_inv)
        b_new = ps.log() - torch.einsum('nd, dc, nc -> n', mu, cov_inv, mu) / 2
            
        test_logits = 100. * test_features_new.float() @ clip_weights_new.float() + alpha * (test_features_new.float() @ W_new + b_new) 
        new_acc = cls_acc(test_logits, test_labels_new)
        
        print("seed: %s, base_acc: %s \t new_acc: %s" % (cfg["seed"], base_acc, new_acc))
        
    return base_acc, new_acc

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
    
    base_accs = []
    new_accs = []
    for seed in [1, 2, 3]:
        cfg["seed"] = seed
        random.seed(seed)
        torch.manual_seed(seed) 
        
        print("Preparing dataset.")
        global train_loader_F, train_loader_cache
        global test_features, test_labels
        global val_features, val_labels
        global test_features_new, test_labels_new
        if cfg['dataset'] != "imagenet":
            # read base dataset.
            cfg["subsample_classes"] = "base"
            dataset = build_dataset(cfg, cfg['dataset'], cfg['root_path'], cfg['shots']) 
            
            train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
            train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True) 

            test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)
            val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)

            test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
            val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)
            
            clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model.float())
            
            # read new_dataset.
            cfg["subsample_classes"] = "new"
            dataset = build_dataset(cfg, cfg['dataset'], cfg['root_path'], cfg['shots']) 
            test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)
            test_features_new, test_labels_new = pre_load_features(cfg, "test", clip_model, test_loader)
            clip_weights_new = clip_classifier(dataset.classnames, dataset.template, clip_model.float())
        else:
            # read base dataset.
            cfg["subsample_classes"] = "base"
            dataset = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)

            train_loader_F = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=True)
            train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)
            
            test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
            val_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
            
            test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
            val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)    
              
            clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model.float())
            
            # read new dataset.
            cfg["subsample_classes"] = "new"
            dataset = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
            test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
            test_features_new, test_labels_new = pre_load_features(cfg, "test", clip_model, test_loader)
            test_labels_new = test_labels_new - 500
            clip_weights_new = clip_classifier(dataset.classnames, dataset.template, clip_model.float())    

        base_acc, new_acc = run(cfg, train_loader_cache, clip_weights, clip_weights_new, clip_model)
        base_accs.append(base_acc)
        new_accs.append(new_acc)
        
    print("Evaluate on dataset:", cfg['dataset'])
    print("Evaluate on seed [1, 2, 3]")
    print("Base acc:", base_accs)
    print("Base mean:", torch.tensor(base_accs).mean())
    print("New acc:", new_accs)
    print("New mean:", torch.tensor(new_accs).mean())
    
if __name__ == '__main__':
    main()