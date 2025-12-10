import os
import pandas as pd
from pathlib import Path

def get_bbox(resolution, person_id, base_path):
    
    path = base_path / resolution / person_id / 'S001/L1/E01'
    
    bbox_dict = {}
    for cfile in os.listdir(path):
        if '.txt' in cfile:
            txt_path = os.path.join(path, cfile)
            txt_df = pd.read_csv(txt_path, sep='\t', header=None, skiprows=[0,1,2,3,4,5,6])
            bbox = txt_df.iloc[0].values
            
            cname = cfile.replace('.txt', '')
            bbox_dict[cname] = bbox
    
    return bbox_dict