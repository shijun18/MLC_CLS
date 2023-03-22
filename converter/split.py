import os
import random
import shutil

input_dir = '/staff/shijun/torch_projects/MLC_CLS/dataset/MLC_v2/MLCnewdata'
save_dir = '/staff/shijun/torch_projects/MLC_CLS/dataset/MLC_v2'

train_dir = os.path.join(save_dir,'train')
test_dir = os.path.join(save_dir,'test')

os.makedirs(train_dir)
os.makedirs(test_dir)

for subdir in os.scandir(input_dir):
    train_subdir = os.path.join(train_dir,subdir.name)
    test_subdir = os.path.join(test_dir,subdir.name)

    os.makedirs(train_subdir)
    os.makedirs(test_subdir)  
    
    sample_list= os.listdir(subdir.path)
    random.shuffle(sample_list)

    retain = int(len(sample_list)*0.2)
    train_sample = sample_list[:-retain]

    for sample in os.scandir(subdir.path):
        if sample.name in train_sample:
            shutil.move(sample.path,os.path.join(train_subdir,sample.name))
        else:
            shutil.move(sample.path,os.path.join(test_subdir,sample.name))

