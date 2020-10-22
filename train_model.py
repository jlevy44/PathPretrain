import fire
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets as Datasets
import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import pandas as pd
from models import generate_model, ModelTrainer
from PIL import Image
from pathflowai.utils import load_sql_df
import torch.nn as nn
import kornia.augmentation as K, kornia.geometry.transform as G
from datasets import NPYDataset, PickleDataset

class Reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x.view(x.shape[0],-1)

def generate_transformers(image_size=224, resize=256, mean=[], std=[], include_jitter=False):
    train_transform = [transforms.Resize((resize,resize))]
    if include_jitter:
        train_transform.append(transforms.ColorJitter(brightness=0.4,
                                            contrast=0.4, saturation=0.4, hue=0.1))
    train_transform.extend([transforms.RandomHorizontalFlip(p=0.5),
           transforms.RandomVerticalFlip(p=0.5),
           transforms.RandomRotation(90),
           transforms.RandomResizedCrop((image_size,image_size)),
           transforms.ToTensor(),
           transforms.Normalize(mean if mean else [0.5, 0.5, 0.5],
                                std if std else [0.1, 0.1, 0.1])
           ])
    train_transform=transforms.Compose(train_transform)
    val_transform = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.CenterCrop((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean if mean else [0.5, 0.5, 0.5],
                             std if std else [0.1, 0.1, 0.1])
    ])
    normalization_transform = transforms.Compose([transforms.Resize((resize,resize)),
                                                  transforms.CenterCrop(
                                                      (image_size,image_size)),
                                                  transforms.ToTensor()])
    return {'train': train_transform, 'val': val_transform, 'test': val_transform, 'norm': normalization_transform}

def generate_kornia_transforms(image_size=224, resize=256, mean=[], std=[], include_jitter=False):
    mean=torch.tensor(mean) if mean else torch.tensor([0.5, 0.5, 0.5])
    std=torch.tensor(std) if std else torch.tensor([0.1, 0.1, 0.1])
    if torch.cuda.is_available():
        mean=mean.cuda()
        std=std.cuda()
    train_transforms=[G.Resize((resize,resize))]
    if include_jitter:
        train_transforms.append(K.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1))
    train_transforms.extend([K.RandomHorizontalFlip(p=0.5),
           K.RandomVerticalFlip(p=0.5),
           K.RandomRotation(90),
           K.RandomResizedCrop((image_size,image_size)),
           K.Normalize(mean,std)
           ])
    val_transforms=[G.Resize((resize,resize)),
           K.CenterCrop((image_size,image_size)),
           K.Normalize(mean,std)
           ]
    transforms=dict(train=nn.Sequential(*train_transforms),
                val=nn.Sequential(*val_transforms))
    if torch.cuda.is_available():
        for k in transforms:
            transforms[k]=transforms[k].cuda()
    return transforms

class SegmentationTransform(nn.Module):
    def __init__(self,resize,image_size,mean,std,include_jitter=False,Set="train"):
        super().__init__()
        self.resize=G.Resize((resize,resize),align_corners=False)
        self.mask_resize=G.Resize((resize,resize),interpolation='nearest',align_corners=False)
        self.jit=K.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1) if include_jitter else (lambda x: x)
        # self.rotations=nn.ModuleList([
        #        K.augmentation.RandomAffine([-90., 90.], [0., 0.15], [0.5, 1.5], [0., 0.15])
        #        # K.RandomHorizontalFlip(p=0.5),
        #        # K.RandomVerticalFlip(p=0.5),
        #        # K.RandomRotation(90),#K.RandomResizedCrop((image_size,image_size),interpolation="nearest")
        #        ])
        # self.rotations_mask=nn.ModuleList([
        #        K.augmentation.RandomAffine([-90., 90.], [0., 0.15], [0.5, 1.5], [0., 0.15],resample="NEAREST")
        #        ])
        self.affine=K.augmentation.RandomAffine([-90., 90.], [0., 0.15], None, [0., 0.15])
        self.affine_mask=K.augmentation.RandomAffine([-90., 90.], [0., 0.15], None, [0., 0.15],resample="NEAREST")
        self.normalize=K.Normalize(mean,std)
        self.crop,self.mask_crop=K.CenterCrop((image_size,image_size)),K.CenterCrop((image_size,image_size),resample="NEAREST")
        self.Set=Set

    def forward(self,input,mask):
        mask=mask.unsqueeze(1).float()#torch.cat([mask.unsqueeze(1)]*3,1)
        if self.Set=='train':
            img=self.jit(self.resize(input))
            mask_out=self.mask_resize(mask)
            img=self.affine(img)
            mask_out=self.affine_mask(mask_out)
            # for rotation in self.rotations: img=rotation(img)
            img=self.normalize(img)
            # for i in range(len(self.rotations_mask)): mask_out=self.rotations_mask[i](mask_out,self.rotations[i]._params)
        else:
            img=self.normalize(self.crop(self.resize(input)))
            mask_out=self.mask_crop(self.mask_resize(mask))
        return img,mask_out.squeeze(1).long()#[:,0,...]

def generate_kornia_segmentation_transforms(image_size=224, resize=256, mean=[], std=[], include_jitter=False):  # add this then IoU metric
    mean=torch.tensor(mean) if mean else torch.tensor([0.5, 0.5, 0.5])
    std=torch.tensor(std) if std else torch.tensor([0.1, 0.1, 0.1])
    transforms={k:SegmentationTransform(resize,image_size,mean,std,include_jitter=False,Set=k) for k in ['train','val']}
    if torch.cuda.is_available():
        for k in transforms:
            transforms[k]=transforms[k].cuda()
    return transforms

def train_model(inputs_dir='inputs_training',
                learning_rate=1e-4,
                n_epochs=300,
                crop_size=224,
                resize=256,
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=2,
                architecture='resnet50',
                batch_size=32,
                predict=False,
                model_save_loc='saved_model.pkl',
                predictions_save_path='predictions.pkl',
                predict_set='test',
                verbose=False,
                class_balance=True,
                extract_embeddings="",
                extract_embeddings_df="",
                embedding_out_dir="./",
                gpu_id=0,
                checkpoints_dir="checkpoints",
                tensor_dataset=False,
                pickle_dataset=False,
                label_map=dict(),
                semantic_segmentation=False,
                save_metric="loss",
                custom_dataset=None,
                save_predictions=True
                ):
    assert save_metric in ['loss','f1']
    if extract_embeddings: assert predict, "Must be in prediction mode to extract embeddings"
    if tensor_dataset: assert not pickle_dataset, "Cannot have pickle and tensor classes activated"
    if semantic_segmentation and custom_dataset is None: assert tensor_dataset==True, "For now, can only perform semantic segmentation with TensorDataset"
    torch.cuda.set_device(gpu_id)
    transformers=generate_transformers if not tensor_dataset else generate_kornia_transforms
    if semantic_segmentation: transformers=generate_kornia_segmentation_transforms
    transformers = transformers(
        image_size=crop_size, resize=resize, mean=mean, std=std)
    if not extract_embeddings:
        if custom_dataset is not None:
            assert predict
            datasets={}
            datasets['custom']=custom_dataset
            predict_set='custom'
        else:
            if tensor_dataset:
                datasets = {x: torch.load(os.path.join(inputs_dir,f"{x}_data.pth")) for x in ['train','val']}
                for k in datasets:
                    if len(datasets[k].tensors[1].shape)>1 and not semantic_segmentation: datasets[k]=TensorDataset(datasets[k].tensors[0],datasets[k].tensors[1].flatten())
            elif pickle_dataset:
                datasets = {x: PickleDataset(os.path.join(inputs_dir,f"{x}_data.pkl"),transformers[x],label_map) for x in ['train','val']}
            else:
                datasets = {x: Datasets.ImageFolder(os.path.join(
                    inputs_dir, x), transformers[x]) for x in ['train', 'val', 'test']}

        dataloaders = {x: DataLoader(
            datasets[x], batch_size=batch_size, shuffle=(x == 'train')) for x in datasets}

    model = generate_model(architecture,
                           num_classes,
                           semantic_segmentation=semantic_segmentation)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer_opts = dict(name='adam',
                          lr=learning_rate,
                          weight_decay=1e-4)

    scheduler_opts = dict(scheduler='warm_restarts',
                          lr_scheduler_decay=0.5,
                          T_max=10,
                          eta_min=5e-8,
                          T_mult=2)

    trainer = ModelTrainer(model,
                           n_epochs,
                           None if predict else dataloaders['val'],
                           optimizer_opts,
                           scheduler_opts,
                           loss_fn='dice' if (semantic_segmentation and not class_balance) else 'ce',
                           checkpoints_dir=checkpoints_dir,
                           tensor_dataset=tensor_dataset,
                           transforms=transformers,
                           semantic_segmentation=semantic_segmentation,
                           save_metric=save_metric)

    if not predict:

        if class_balance:
            trainer.add_class_balance_loss(datasets['train'].targets if not tensor_dataset else datasets['train'].tensors[1].numpy().flatten())

        trainer, min_val_loss_f1, best_epoch=trainer.fit(dataloaders['train'],verbose=verbose)

        torch.save(trainer.model.state_dict(), model_save_loc)

        return trainer.model

    else:
        # assert not tensor_dataset, "Only ImageFolder and NPYDatasets allowed"

        trainer.model.load_state_dict(torch.load(model_save_loc,map_location=f"cuda:{gpu_id}" if gpu_id>=0 else "cpu"))

        if extract_embeddings and extract_embeddings_df:
            assert not semantic_segmentation, "Semantic Segmentation not implemented for whole slide segmentation"
            trainer.model=nn.Sequential(trainer.model.features,Reshape())
            patch_info=load_sql_df(extract_embeddings_df,resize)
            dataset=NPYDataset(patch_info,extract_embeddings,transformers["test"],tensor_dataset)
            dataset.embed(trainer.model,batch_size,embedding_out_dir)
            exit()

        Y = dict()

        Y['pred'],Y['true'] = trainer.predict(dataloaders[predict_set])

        # Y['true'] = datasets[predict_set].targets

        if save_predictions: torch.save(Y, predictions_save_path)

        return Y


if __name__ == '__main__':
    fire.Fire(train_model)
