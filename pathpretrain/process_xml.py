import numpy as np
from scipy.interpolate import splprep, splev
import xmltodict as xd
import pandas as pd
import tqdm

def fit_spline(pts):
    try:
        okay = np.where(np.abs(np.diff(pts,axis=0)).sum(1) > 0)[0]
        pts = np.vstack([pts[okay], pts[-1]])#, pts[0]
        tck, u = splprep(pts.T, u=None, s=0.0, per=1)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)
        pts=np.vstack((x_new, y_new)).T
    except:
        pass
    return pts

def process_xml(xml,
                compression=1,
                include_labels=[],
                spline_fit=True,
                transpose=True,
                return_dot=True,
                return_contour=True):
    with open(xml,"rb") as f:
        d=xd.parse(f)
    contour_df=None
    dot_annotations=None
    contours=[]
    dots=[]
    lbls=[]
    annotations=d['ASAP_Annotations']["Annotations"]
    if annotations:
        for i,annotation in enumerate(annotations["Annotation"]):
            if return_contour and annotation['@Type'] in ['Polygon','Spline']:
                try:
                    lbl=annotation["@PartOfGroup"]
                    contour=np.array([(float(coord["@X"]),float(coord["@Y"])) for coord in annotation["Coordinates"]["Coordinate"]])
                    contours.append(contour)
                    lbls.append(lbl)
                except:
                    print(xml,i,annotation.keys())

            try:
                if return_dot and annotation["@Type"]=="Dot":
                    lbl=annotation.get("@PartOfGroup","")
                    dots.append((float(annotation["Coordinates"]["Coordinate"]["@X"]),
                                     float(annotation["Coordinates"]["Coordinate"]["@Y"]),
                                     lbl))
            except:
                print(annotation)

    if contours:
        contour_df=pd.DataFrame(pd.Series(contours,name='contours'))
        contour_df['contours']=contour_df['contours'].map(lambda x:x/compression)
        if transpose: contour_df['contours']=contour_df['contours'].map(lambda x:x[:,[1,0]])
        contour_df['xmin']=contour_df['contours'].map(lambda x: x[:,0].min())
        contour_df['xmax']=contour_df['contours'].map(lambda x: x[:,0].max())
        contour_df['ymin']=contour_df['contours'].map(lambda x: x[:,1].min())
        contour_df['ymax']=contour_df['contours'].map(lambda x: x[:,1].max())
        contour_df['xmean']=contour_df['contours'].map(lambda x: x[:,0].mean())
        contour_df['ymean']=contour_df['contours'].map(lambda x: x[:,1].mean())
        contour_df['lbl']=[lbl.lower() for lbl in lbls]

        if spline_fit: contour_df.loc[:,'contours']=contour_df['contours'].map(fit_spline)

    if dots:
        dot_annotations=pd.DataFrame(dots,columns=['x','y','lbl']) if dots else None
        dot_annotations.loc[:,['x','y']]=np.round(dot_annotations.loc[:,['x','y']]/compression).astype(int)
        if transpose: dot_annotations.loc[:,['x','y']]=dot_annotations.loc[:,['y','x']]

    if contours and include_labels:
        contour_df=contour_df[contour_df['lbl'].isin(include_labels)]

    return dict(contour=contour_df,
                dot=dot_annotations)
