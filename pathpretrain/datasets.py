import torch
import os
import pickle
import tifffile
from PIL import Image
import tqdm
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
from .utils import load_image

class NPYDataset(Dataset):
    def __init__(self, patch_info, npy_file, transform, tensor_dataset=False):
        self.ID=os.path.basename(npy_file).replace(".npy","").replace(".tiff","").replace(".tif","").replace(".svs","")
        self.patch_info=patch_info.loc[patch_info["ID"]==self.ID].reset_index()
        self.X=load_image(npy_file)
        self.to_pil=lambda x: Image.fromarray(x)
        self.transform=transform
        self.tensor_dataset=tensor_dataset

    def __getitem__(self,i):
        x,y,patch_size=self.patch_info.loc[i,["x","y","patch_size"]]
        img=self.X[x:x+patch_size,y:y+patch_size]
        return self.transform(self.to_pil(img)) if not self.tensor_dataset else torch.tensor(img),torch.tensor([-1])

    def __len__(self):
        return self.patch_info.shape[0]

    def embed(self,model,batch_size,out_dir):
        Z=[]
        dataloader=DataLoader(self,batch_size=batch_size,shuffle=False)
        n_batches=len(self)//batch_size
        with torch.no_grad():
            for i,(X,y) in tqdm.tqdm(enumerate(dataloader),total=n_batches):
                if torch.cuda.is_available(): X=X.cuda()
                if self.tensor_dataset: X = self.transform(X)
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
        self.aux_data=self.data.get("z",None)
        self.has_aux=(self.aux_data is not None)
        if self.has_aux and isinstance(self.aux_data,pd.DataFrame): self.aux_data=self.aux_data.values
        if self.has_aux: self.n_aux_features=self.aux_data.shape[1]
        self.transform=transform
        self.to_pil=lambda x: Image.fromarray(x)
        self.label_map=label_map
        if self.label_map:
            self.targets=pd.Series(self.targets).map(lambda x: self.label_map.get(x,-1)).values
            if -1 in self.targets:
                remove_bool=(self.targets!=-1)
                self.targets=self.targets[remove_bool]
                self.X=pd.Series(self.X).iloc[remove_bool].tolist()
                if self.has_aux: self.aux_data=self.aux_data[remove_bool]
        self.length=len(self.X)


    def __getitem__(self,idx):
        items=(self.transform(self.to_pil(self.X[idx])), torch.tensor(self.targets[idx]).long())
        if self.has_aux: items+=(torch.tensor(self.aux_data[idx]).float(),)
        return items

    def __len__(self):
        return self.length

class NPYRotatingStack(Dataset):
    def __init__(self, patch_dir, transform, sample_frac=1., sample_every=0, target_col={'old_y_true':'y_true'},npy_rotate_sets_pkl="",Set=""):
        self.npy_rotate_sets_pkl=npy_rotate_sets_pkl
        if npy_rotate_sets_pkl:
            self.patch_npy=pd.read_pickle(self.npy_rotate_sets_pkl)
            self.patch_pkl=self.patch_npy[self.patch_npy['Set']==Set]['pkl'].values
            self.patch_npy=self.patch_npy[self.patch_npy['Set']==Set]['npy'].values
        else:
            self.patch_npy=np.array(glob.glob(os.path.join(patch_dir,"*.npy")))
            self.patch_pkl=np.vectorize(lambda x: x.replace(".npy",".pkl"))(self.patch_npy)
        self.sample_every=sample_every
        self.sample_frac=sample_frac
        if self.sample_frac==1: self.sample_every=0
        self.target_col=list(target_col.items())[0]
        self.ref_index=None # dictionary
        self.data={}
        self.cache_npy=None # dictionary keys
        self.to_pil=lambda x: Image.fromarray(x)
        self.transform=transform
        assert self.target_col[1]=='y_true'
        self.targets=np.hstack([pd.read_pickle(pkl)[self.target_col[0]].values for pkl in self.patch_pkl])
        self.load_image_annot()

    def load_image_annot(self):
        if self.sample_frac<1.:
            idx=np.arange(len(self.patch_npy))
            idx=np.random.choice(idx,int(self.sample_frac*len(idx)))
            patch_npy=self.patch_npy[idx]
            patch_pkl=self.patch_pkl[idx]
            remove_npy=np.setdiff1d(self.patch_npy,patch_npy)
            for npy in remove_npy:
                if isinstance(self.cache_npy,type(None))==False and npy not in self.cache_npy:
                    del self.data[npy]
            new_data={npy:(dict(patches=load_image(npy),
                               patch_info=pd.read_pickle(pkl)) if (self.cache_npy is None or (npy not in self.cache_npy if self.cache_npy is not None else False)) else self.data[npy]) for npy,pkl in zip(patch_npy,patch_pkl)}
            self.data.clear()
            self.data=new_data
            self.cache_npy=sorted(list(self.data.keys()))
        else:
            self.data={npy:dict(patches=load_image(npy),
                               patch_info=pd.read_pickle(pkl)) for npy,pkl in zip(self.patch_npy,self.patch_pkl)}
            self.cache_npy=sorted(self.patch_npy)
        self.ref_index=np.vstack([np.array(([i]*self.data[npy]['patch_info'].shape[0],list(range(self.data[npy]['patch_info'].shape[0])))).T for i,npy in enumerate(self.cache_npy)])
        for npy in self.data: self.data[npy]['patch_info'][self.target_col[1]]=self.data[npy]['patch_info'][self.target_col[0]]
        self.length=self.ref_index.shape[0]

    def __getitem__(self,idx):
        i,j=self.ref_index[idx]
        npy=self.cache_npy[i]
        X=self.data[npy]['patches'][j]
        y=torch.LongTensor([self.data[npy]['patch_info'].iloc[j][self.target_col[0]]])
        X=self.transform(self.to_pil(X))
        return X, y

    def __len__(self):
        return self.length
