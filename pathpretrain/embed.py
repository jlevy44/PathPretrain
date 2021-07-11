import fire, os, torch, tqdm, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from .train_model import train_model, generate_transformers, generate_kornia_transforms
from .utils import load_image

class CustomDataset(Dataset):
    def __init__(self, patch_info, npy_file, transform, image_stack=False):
        self.X=load_image(npy_file)
        self.patch_info=pd.read_pickle(patch_info)
        self.xy=self.patch_info[['x','y']].values
        self.patch_size=self.patch_info['patch_size'].iloc[0]
        self.length=self.patch_info.shape[0]
        self.transform=transform
        self.to_pil=lambda x: Image.fromarray(x)
        self.ID=os.path.basename(npy_file).replace(".npy","")

    def __getitem__(self,i):
        x,y=self.xy[i]
        X=self.X[i] if image_stack else self.X[x:(x+patch_size),y:(y+patch_size)]
        return self.transform(self.to_pil(X))

    def __len__(self):
        return self.length

    def embed(self,model,batch_size,out_dir):
        Z=[]
        dataloader=DataLoader(self,batch_size=batch_size,shuffle=False)
        n_batches=len(self)//batch_size
        with torch.no_grad():
            for i,X in tqdm.tqdm(enumerate(dataloader),total=n_batches):
                if torch.cuda.is_available(): X=X.cuda()
                z=model(X).detach().cpu().numpy()
                Z.append(z)
        Z=np.vstack(Z)
        torch.save(dict(embeddings=Z,patch_info=self.patch_info),os.path.join(out_dir,f"{self.ID}.pkl"))

def generate_embeddings(patch_info_file="",
                        image_file="",
                        model_save_loc="",
                        architecture="resnet50",
                        num_classes=4,
                        gpu_id=-1,
                        crop_size=224,
                        resize=256,
                        mean=[0.5, 0.5, 0.5],
                        std=[0.1, 0.1, 0.1],
                        image_stack=False):

    os.makedirs("cnn_embeddings",exist_ok=True)
    train_model(model_save_loc=model_save_loc,
                extract_embeddings=True,
                num_classes=num_classes,
                predict=True,
                embedding_out_dir="cnn_embeddings/",
                custom_dataset=CustomDataset(patch_info_file,
                                             image_file,
                                             generate_transformers(image_size=crop_size,
                                                                    resize=resize,
                                                                    mean=mean,
                                                                    std=std)['test'],
                                             image_stack
                                             ),
                gpu_id=gpu_id)

def main():
    fire.Fire(generate_embeddings)

if __name__=="__main__":
    main()
