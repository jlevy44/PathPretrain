import torch
import os
import pickle

from PIL import Image
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader

class NPYDataset(Dataset):
    def __init__(self, patch_info, npy_file, transform, tensor_dataset):
        self.ID=os.path.basename(npy_file).split('.')[0]
        self.patch_info=patch_info.loc[patch_info["ID"]==self.ID].reset_index()
        self.X=np.load(npy_file)
        self.to_pil=lambda x: Image.fromarray(x)
        self.transform=transform
        self.tensor_dataset=tensor_dataset

    def __getitem__(self,i):
        x,y,patch_size=self.patch_info.loc[i,["x","y","patch_size"]]
        img=self.X[x:x+patch_size,y:y+patch_size]
        return self.transform(self.to_pil(img)) if not self.tensor_dataset else torch.tensor(img)

    def __len__(self):
        return self.patch_info.shape[0]

    def embed(self,model,batch_size,out_dir):
        Z=[]
        dataloader=DataLoader(self,batch_size=batch_size,shuffle=False)
        n_batches=len(self)//batch_size
        with torch.no_grad():
            for i,X in enumerate(dataloader):
                if torch.cuda.is_available(): X=X.cuda()
                if torch.tensor_dataset: X = self.transform(X)
                z=model(X).detach().cpu().numpy()
                Z.append(z)
                print(f"Processed batch {i}/{n_batches}")
        Z=np.vstack(Z)
        torch.save(dict(embeddings=Z,patch_info=self.patch_info),os.path.join(out_dir,f"{self.ID}.pkl"))
        print("Embeddings saved")
        quit()

class PickleDataset(Dataset):
    def __init__(self, pkl, transform, label_map):
        self.data=pickle.load(open(pkl,'rb'))
        self.X,self.targets=self.data['X'],self.data['y']
        self.transform=transform
        self.to_pil=lambda x: Image.fromarray(x)
        self.label_map=label_map
        if self.label_map:
            self.targets=pd.Series(self.targets).map(lambda x: self.label_map.get(x,-1)).values
            if -1 in self.targets:
                remove_bool=(self.targets!=-1)
                self.targets=self.targets[remove_bool]
                self.X=pd.Series(self.X).iloc[remove_bool].tolist()
        self.length=len(self.X)


    def __getitem__(self,idx):
        return self.transform(self.to_pil(self.X[idx])), torch.tensor(self.targets[idx]).long()

    def __len__(self):
        return self.length
