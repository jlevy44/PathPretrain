import tqdm
import cv2
import os
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import tifffile

# Section taken from: https://github.com/jlevy44/PathFlowAI/blob/master/pathflowai/utils.py

# from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import label as scilabel, distance_transform_edt
import scipy.ndimage as ndimage
from skimage import morphology as morph
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
from skimage.filters import threshold_otsu, rank
from skimage.morphology import convex_hull_image, remove_small_holes
from skimage import measure


def filter_grays(rgb, tolerance=15, output_type="bool"):
    """
    https://github.com/deroneriksson/python-wsi-preprocessing/blob/master/deephistopath/wsi/filter.py
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.
    Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).
    Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    (h, w, c) = rgb.shape
    rgb = rgb.astype(np.int)
    rg_diff = np.abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = np.abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = np.abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def label_objects(img,
                  otsu=True,
                  min_object_size=100000,
                  threshold=240,
                  connectivity=8,
                  kernel=61,
                  keep_holes=False,
                  max_hole_size=0,
                  gray_before_close=False,
                  blur_size=0):
    I=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray_mask=filter_grays(img, output_type="bool")
    if otsu: threshold = threshold_otsu(I)
    BW = (I<threshold).astype(bool)
    if gray_before_close: BW=BW&gray_mask
    if kernel>0: BW = morph.binary_closing(BW, morph.disk(kernel))#square
    if not gray_before_close: BW=BW&gray_mask
    if blur_size: BW=(cv2.blur(BW.astype(np.uint8), (blur_size,blur_size))==1)
    labels = scilabel(BW)[0]
    labels=morph.remove_small_objects(labels, min_size=min_object_size, connectivity = connectivity, in_place=True)
    if not keep_holes and max_hole_size:
        BW=morph.remove_small_objects(labels==0, min_size=max_hole_size, connectivity = connectivity, in_place=True)==False#remove_small_holes(labels,area_threshold=max_hole_size, connectivity = connectivity, in_place=True)>0
    elif keep_holes:
        BW=labels>0
    else:
        BW=fill_holes(labels)
    labels = scilabel(BW)[0]
    return(BW!=0),labels


def generate_tissue_mask(arr,
                            compression=8,
                            otsu=False,
                            threshold=220,
                            connectivity=8,
                            kernel=61,
                            min_object_size=100000,
                            return_convex_hull=False,
                            keep_holes=False,
                            max_hole_size=0,
                            gray_before_close=False,
                            blur_size=0):
    img=cv2.resize(arr,None,fx=1/compression,fy=1/compression,interpolation=cv2.INTER_CUBIC)
    WB, lbl=label_objects(img, otsu=otsu, min_object_size=min_object_size, threshold=threshold, connectivity=connectivity, kernel=kernel,keep_holes=keep_holes,max_hole_size=max_hole_size, gray_before_close=gray_before_close,blur_size=blur_size)
    if return_convex_hull:
        for i in range(1,lbl.max()+1):
            WB=WB+convex_hull_image(lbl==i)
        WB=WB>0
    WB=cv2.resize(WB.astype(np.uint8),arr.shape[:2][::-1],interpolation=cv2.INTER_CUBIC)>0
    return WB

######################################


def deduplicate_images(image_list):
    image_list=pd.Series(image_list) # if X is a pandas series containing images for individual elements
    shapes_=np.array([x.shape[:2] for x in image_list]) # get shapes
    d_mat=(euclidean_distances(shapes_)+np.eye(len(shapes_))) # coarse shape matching first
    d_mat[np.tril_indices(len(d_mat))]=1
    d_mat=d_mat==0
    idxs=np.where(d_mat)
    same=[]
    hashes=image_list.map(lambda x: cv2.resize(cv2.cvtColor(x,cv2.COLOR_RGB2GRAY),None,fx=1/compression,fy=1/compression)) # hash to reduce compute time; compression vs accuracy search
    for i,j in tqdm.tqdm(zip(*idxs),total=d_mat.sum()): # search through all image pairs with matching shapes and look for complete alignment with hashes
        if (hashes.iloc[i]==hashes.iloc[j]).mean()==1:
            same.append((i,j)) # update information on matching images
    G=nx.Graph()
    G.add_edges_from(same)
    remove=[]
    for comp in nx.connected_components(G):
        remove.extend(list(comp)[1:])
    return image_list.drop(remove).tolist()

def load_image(image_file, check_size=False, mmap_mode=None):
    img_ext=os.path.splitext(image_file)
    if img_ext[-1]==".npy":
        image=np.load(image_file, mmap_mode=mmap_mode)
    elif img_ext[-1] in [".svs",".tif",".tiff",".png"]:
        if check_size:
            import openslide
            slide=openslide.open_slide(image_file)
        image=tifffile.imread(image_file, aszarr=mmap_mode is not None)
        if mmap_mode is not None:
            import zarr
            image=zarr.open(image, mode=mmap_mode)
        if check_size and (not (int(slide.properties.get('aperio.AppMag',40))==20 or int(slide.properties.get('openslide.objective-power',40))==20)):
            image = cv2.resize(image,None,fx=1/2,fy=1/2,interpolation=cv2.INTER_CUBIC)
    else:
        raise NotImplementedError
    return image
