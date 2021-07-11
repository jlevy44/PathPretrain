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
                transpose=True):
    with open(xml,"rb") as f:
            d=xd.parse(f)
    cells=[]
    cell_dot=[]
    lbls=[]
    for i,annotation in enumerate(d['ASAP_Annotations']["Annotations"]["Annotation"]):
        if annotation['@Type'] in ['Polygon','Spline']:
            try:
                lbl=annotation["@PartOfGroup"]
                contour=np.array([(float(coord["@X"]),float(coord["@Y"])) for coord in annotation["Coordinates"]["Coordinate"]])
                cells.append(contour)
                lbls.append(lbl)
            except:
                print(xml,i,annotation.keys())

        if annotation["@Type"]=="Dot":
            lbl=annotation.get("@PartOfGroup","")
            cell_dot.append((float(annotation["Coordinates"]["Coordinate"]["@X"]),
                             float(annotation["Coordinates"]["Coordinate"]["@Y"]),
                             lbl))

    contour_df=pd.DataFrame(pd.Series(cells,name='contours'))
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

    if cell_dot:
        dot_annotations=pd.DataFrame(cell_dot,columns=['x','y','lbl']) if cell_dot else None
        dot_annotations.loc[:,['x','y']]=np.round(dot_annotations.loc[:,['x','y']]/compression).astype(int)
        if transpose: dot_annotations.loc[:,['x','y']]=dot_annotations.loc[:,['y','x']]
    else:
        dot_annotations=None

    if include_labels:
        contour_df=contour_df[contour_df['lbl'].isin(include_labels)]

    return dict(contour=contour_df,
                dot=dot_annotations)
