import fire
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets as Datasets
import sys
import os
import numpy as np
import pandas as pd
from models import generate_model, ModelTrainer


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


def train_model(inputs_dir='inputs_training',
                learning_rate=1e-4,
                n_epochs=300,
                crop_size=224,
                resize=256,
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=3,
                architecture='resnet50',
                batch_size=32,
                predict=False,
                model_save_loc='saved_model.pkl',
                predictions_save_path='predictions.pkl',
                predict_set='test',
                verbose=False,
                class_balance=True
                ):
    transformers = generate_transformers(
        image_size=crop_size, resize=resize, mean=mean, std=std)
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
                           dataloaders['val'],
                           optimizer_opts,
                           scheduler_opts,
                           loss_fn='ce')

    if class_balance:
        trainer.add_class_balance_loss(datasets['train'].targets)

    if not predict:

        trainer, min_val_loss, best_epoch=trainer.fit(dataloaders['train'],verbose=verbose)

        torch.save(trainer.model.state_dict(), model_save_loc)

    else:

        trainer.model.load_state_dict(torch.load(model_save_loc))

        Y = dict()

        Y['pred'],Y['true'] = trainer.predict(dataloaders[predict_set])

        # Y['true'] = datasets[predict_set].targets

        torch.save(Y, predictions_save_path)


if __name__ == '__main__':
    fire.Fire(train_model)
