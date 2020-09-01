import fire
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets as Datasets
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from models import generate_model, ModelTrainer
from PIL import Image
from pathflowai.utils import load_sql_df
import torch.nn as nn
import kornia.augmentation as K, kornia.geometry.transform as G

class Reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x.view(x.shape[0],-1)

class NPYDataset(Dataset):
    def __init__(self, patch_info, npy_file, transform):
        self.ID=os.path.basename(npy_file).split('.')[0]
        self.patch_info=patch_info.loc[patch_info["ID"]==self.ID].reset_index()
        self.X=np.load(npy_file)
        self.to_pil=lambda x: Image.fromarray(x)
        self.transform=transform

    def __getitem__(self,i):
        x,y,patch_size=self.patch_info.loc[i,["x","y","patch_size"]]
        return self.transform(self.to_pil(self.X[x:x+patch_size,y:y+patch_size]))

    def __len__(self):
        return self.patch_info.shape[0]

    def embed(self,model,batch_size,out_dir):
        Z=[]
        dataloader=DataLoader(self,batch_size=batch_size,shuffle=False)
        n_batches=len(self)//batch_size
        with torch.no_grad():
            for i,X in enumerate(dataloader):
                if torch.cuda.is_available():
                    X=X.cuda()
                z=model(X).detach().cpu().numpy()
                Z.append(z)
                print(f"Processed batch {i}/{n_batches}")
        Z=np.vstack(Z)
        torch.save(dict(embeddings=Z,patch_info=self.patch_info),os.path.join(out_dir,f"{self.ID}.pkl"))
        print("Embeddings saved")
        quit()



def generate_transformers(image_size=224, resize=256, mean=[], std=[], include_jitter=False):

    train_transform = transforms.Compose([
        transforms.Resize(resize)]
        + ([transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1)] if include_jitter else [])
        + [transforms.RandomHorizontalFlip(p=0.5),
           transforms.RandomVerticalFlip(p=0.5),
           transforms.RandomRotation(90),
           transforms.RandomResizedCrop(image_size),
           transforms.ToTensor(),
           transforms.Normalize(mean if mean else [0.5, 0.5, 0.5],
                                std if std else [0.1, 0.1, 0.1])
           ])
    val_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean if mean else [0.5, 0.5, 0.5],
                             std if std else [0.1, 0.1, 0.1])
    ])
    normalization_transform = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(
                                                      image_size),
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
                tensor_dataset=False
                ):
    if extract_embeddings: assert extract_embeddings==predict, "Must be in prediction mode to extract embeddings"
    torch.cuda.set_device(gpu_id)
    transformers=generate_transformers if not tensor_dataset else generate_kornia_transforms
    transformers = transformers(
        image_size=crop_size, resize=resize, mean=mean, std=std)
    if not extract_embeddings:
        if tensor_dataset:
            datasets = {x: torch.load(os.path.join(inputs_dir,f"{x}_data.pth")) for x in ['train','val']}
        else:
            datasets = {x: Datasets.ImageFolder(os.path.join(
                inputs_dir, x), transformers[x]) for x in ['train', 'val', 'test']}

        dataloaders = {x: DataLoader(
            datasets[x], batch_size=batch_size, shuffle=(x == 'train')) for x in datasets}

    model = generate_model(architecture,
                           num_classes)

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
                           loss_fn='ce',
                           checkpoints_dir=checkpoints_dir,
                           tensor_dataset=tensor_dataset,
                           transforms=transformers)

    if not predict:

        if class_balance:
            trainer.add_class_balance_loss(datasets['train'].targets if not tensor_dataset else datasets['train'].tensors[1].numpy())

        trainer, min_val_loss, best_epoch=trainer.fit(dataloaders['train'],verbose=verbose)

        torch.save(trainer.model.state_dict(), model_save_loc)

    else:
        assert not tensor_dataset, "Only ImageFolder and NPYDatasets allowed"

        trainer.model.load_state_dict(torch.load(model_save_loc))

        if extract_embeddings and extract_embeddings_df:
            trainer.model=nn.Sequential(trainer.model.features,Reshape())
            patch_info=load_sql_df(extract_embeddings_df,resize)
            dataset=NPYDataset(patch_info,extract_embeddings,transformers["test"])
            dataset.embed(trainer.model,batch_size,embedding_out_dir)
            exit()

        Y = dict()

        Y['pred'],Y['true'] = trainer.predict(dataloaders[predict_set])

        # Y['true'] = datasets[predict_set].targets

        torch.save(Y, predictions_save_path)


if __name__ == '__main__':
    fire.Fire(train_model)
