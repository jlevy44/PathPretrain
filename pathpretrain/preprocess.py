import fire
import cv2, os, tqdm, dask
import pandas as pd, numpy as np
import histomicstk as htk
from dask.diagnostics import ProgressBar
from pathflowai.utils import generate_tissue_mask
from .utils import load_image
from itertools import product
from histomicstk.preprocessing.color_normalization.deconvolution_based_normalization import deconvolution_based_normalization


DEFAULT_MASK_PARAMETERS=dict(compression=10,
                            otsu=False,
                            threshold=240,
                            connectivity=8,
                            kernel=5,
                            min_object_size=100000,
                            return_convex_hull=False,
                            keep_holes=False,
                            max_hole_size=6000,
                            gray_before_close=True,
                            blur_size=51)

NORM_PATCH_SIZE=1024

def return_norm_image(img,mask,W_source=None,W_target=None):
    img=deconvolution_based_normalization(
        img, W_source=W_source, W_target=W_target, im_target=None,
        stains=['hematoxylin', 'eosin'], mask_out=~mask,
        stain_unmixing_routine_params={"I_0":215})
    return img

def stain_norm(image, mask, compression, stain_target_parameters, patch_size=1024):
    img_small=cv2.resize(image,None,fx=1/compression,fy=1/compression)
    W_source = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(img_small, 215)
    W_source = htk.preprocessing.color_deconvolution._reorder_stains(W_source)
    W_target = np.load(stain_target_parameters)
    res=[]
    coords=[]
    for i in np.arange(0,image.shape[0]-patch_size,patch_size):
        for j in np.arange(0,image.shape[1]-patch_size,patch_size):
            if mask[i:i+patch_size,j:j+patch_size].mean()>1e-2:
                coords.append((i,j))
                res.append(dask.delayed(return_norm_image)(image[i:i+patch_size,j:j+patch_size],mask[i:i+patch_size,j:j+patch_size],W_source,W_target=W_target))
    with ProgressBar():
        res_returned=dask.compute(*res,scheduler="processes")
    return res_returned, coords

def preprocess(image_file="",
               threshold=0.05,
               save_tissue_mask=False,
               new_mask_parameters=dict(),
               stain_target_parameters="",
               compression=8,
               patch_size=256
               ):

    mask_parameters=DEFAULT_MASK_PARAMETERS
    image=load_image(image_file)
    mask_parameters.update(new_mask_parameters)
    mask=generate_tissue_mask(image,
                             **mask_parameters)
    basename = os.path.splitext(os.path.basename(image_file))[0]
    x_max = image.shape[0]
    y_max = image.shape[1]
    if stain_target_parameters and os.path.exists(stain_target_parameters):
        res_returned, coords=stain_norm(image, mask, compression, stain_target_parameters, NORM_PATCH_SIZE)
        for res_,(i,j) in tqdm.tqdm(zip(res_returned,coords),total=len(coords)):
            image[i:i+NORM_PATCH_SIZE,j:j+NORM_PATCH_SIZE]=res_

    patch_info=pd.DataFrame([[basename,x,y,patch_size,"0"] for x,y in tqdm.tqdm(list(product(range(0,x_max-patch_size+1,patch_size),range(0,y_max-patch_size,patch_size))))],columns=['ID','x','y','patch_size','annotation'])
    patches=np.stack([image[x:x+patch_size,y:y+patch_size] for x,y in tqdm.tqdm(patch_info[['x','y']].values.tolist())])
    include_patches=np.stack([mask[x:x+patch_size,y:y+patch_size] for x,y in tqdm.tqdm(patch_info[['x','y']].values.tolist())]).mean((1,2))>=threshold

    os.makedirs("masks",exist_ok=True)
    os.makedirs("patches",exist_ok=True)
    if save_tissue_mask:
        np.save(f"masks/{basename}.npy",mask)
    np.save(f"patches/{basename}.npy",patches[include_patches])
    patch_info.iloc[include_patches].to_pickle(f"patches/{basename}.pkl")

def main():
    fire.Fire(preprocess)

if __name__=="__main__":
    main()
