from meshtopc import genereate_point_cloud
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

if __name__ == '__main__':
	shapenet_df = pd.read_csv('ShapeNetCorev2.csv')
	paths = natsorted(glob('ShapeNetCorev2/*/*/models/*.obj'))

	for path in tqdm(paths):
		try:
			dirname, classname, filename, _, _ = path.split(os.path.sep)
			split_type = shapenet_df[shapenet_df['modelId']==filename]['split'].item()
			if not os.path.isfile('new_'+dirname+'/'+classname+'/'+split_type+'/'+filename+'.npy'):
				P =  pc_normalize(genereate_point_cloud(path, N=2048))
				full_path  = 'new_'+dirname+'/'+classname+'/'+split_type
				if not os.path.isdir(full_path):
					os.makedirs(full_path)
				full_path  = full_path + '/' + filename
				np.save(full_path, P)
			else:
				try:
					_ = np.load('new_'+dirname+'/'+classname+'/'+split_type+'/'+filename+'.npy', allow_pickle=True)
					if _.shape[1]==3:
						pass
				except:
					print('Removing file')
					os.remove('new_'+dirname+'/'+classname+'/'+split_type+'/'+filename+'.npy')
		except Exception as e:
			print(e)