import os
from PIL import Image
from tqdm import tqdm
import numpy as np

def gen_data(img_path,save_path):
	postfix = ['dose_map','gamma_1mm','gamma_2mm']
	dir_name = ['dose','gamma1mm','gamma2mm']
	dose_path = os.path.join(img_path,dir_name[0])
	gamma1_path = os.path.join(img_path,dir_name[1])
	gamma2_path = os.path.join(img_path,dir_name[2])
	dose_name = os.listdir(dose_path)
	dose_name.sort(key=lambda x:x[:12])
	gamma1_name = os.listdir(gamma1_path)
	gamma1_name.sort(key=lambda x:x[:12])
	gamma2_name = os.listdir(gamma2_path)
	gamma2_name.sort(key=lambda x:x[:12])
	for dose_img,gamma1_img,gamma2_img in zip(dose_name,gamma1_name,gamma2_name):
		print(dose_img,gamma1_img,gamma2_img)
		assert dose_img[:12] == gamma1_img[:12] == gamma2_img[:12]
		dose_item = os.path.join(dose_path,dose_img)
		dose_img_r = Image.open(dose_item).convert('L')
		gamma1_item = os.path.join(gamma1_path,gamma1_img)
		gamma1_img_g = Image.open(gamma1_item).convert('L')
		gamma2_item = os.path.join(gamma2_path,gamma2_img)
		gamma2_img_b = Image.open(gamma2_item).convert('L')
		rgb_img = np.stack([np.array(dose_img_r),np.array(gamma1_img_g),np.array(gamma2_img_b)],axis=-1)
		print(rgb_img.shape)
		rgb_img = Image.fromarray(rgb_img, mode='RGB')
		rgb_img_path = os.path.join(save_path,dose_img[:12] + '.png')
		rgb_img.save(rgb_img_path)

if __name__ == '__main__':
	root_path = '../dataset/raw_data'
	train_path = '../dataset/MLC/train'
	for dose_path in os.scandir(root_path):
		save_path = os.path.join(train_path,dose_path.name)
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		gen_data(dose_path.path,save_path)


