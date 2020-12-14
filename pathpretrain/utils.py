import tqdm,cv2
import pandas as pd, networkx as nx, numpy as np
from sklearn.metrics.pairwise import euclidean_distances


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
