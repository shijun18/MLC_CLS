from genericpath import exists
import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import librosa

def sample_count(input_path,save_path):

    info = []
    for item in os.scandir(input_path):
        info_item = [item.name,len(os.listdir(item.path))]
        info.append(info_item)
    info.sort(key=lambda x:eval(x[0]))
    col = ['class','sample_num']
    csv_file = pd.DataFrame(columns=col,data=info)
    csv_file.to_csv(save_path,index=False)

def size_count(input_path,save_path):
    info = []
    for subdir in os.scandir(input_path):
        info_item = [subdir.name]
        size_list = []
        for item in os.scandir(subdir.path):
            img = Image.open(item.path)
            size_list.append(img.size)
        info_item.append(list(set(size_list)))
        info.append(info_item)
    info.sort(key=lambda x:eval(x[0]))
    col = ['class','size_list']
    csv_file = pd.DataFrame(columns=col,data=info)
    csv_file.to_csv(save_path,index=False)


def cal_mean_std(data_path):
    r_mean = []
    g_mean = []
    b_mean = []

    r_std = []
    g_std = []
    b_std = []

    for item in tqdm(data_path):
        img = Image.open(item).convert('RGB').split()
        r_img = (np.array(img[0].resize((128,128))).astype(np.float32)/255.0).flatten()
        r_mean.append(np.mean(r_img))
        g_img = (np.array(img[1].resize((128,128))).astype(np.float32)/255.0).flatten()
        g_mean.append(np.mean(g_img))
        b_img = (np.array(img[2].resize((128,128))).astype(np.float32)/255.0).flatten()
        b_mean.append(np.mean(b_img))
    
    r_mean = np.mean(r_mean)
    g_mean = np.mean(g_mean)
    b_mean = np.mean(b_mean)

    for item in tqdm(data_path):
        img = Image.open(item).convert('RGB').split()
        r_img = (np.array(img[0].resize((128,128))).astype(np.float32)/255.0).flatten()
        r_std.append(np.mean(np.power(r_img - r_mean,2)))
        g_img = (np.array(img[1].resize((128,128))).astype(np.float32)/255.0).flatten()
        g_std.append(np.mean(np.power(g_img - g_mean,2)))
        b_img = (np.array(img[2].resize((128,128))).astype(np.float32)/255.0).flatten()
        b_std.append(np.mean(np.power(b_img - b_mean,2)))

    r_std = np.sqrt(np.mean(r_std))
    g_std = np.sqrt(np.mean(g_std))
    b_std = np.sqrt(np.mean(b_std))

    print('r mean:%.3f' % r_mean)
    print('r std:%.3f' % r_std)
    print('g mean:%.3f' % g_mean)
    print('g std:%.3f' % g_std)
    print('b mean:%.3f' % b_mean)
    print('b std:%.3f' % b_std)



def cal_mean_std_single(data_path):
    l_mean = []
    l_std = []

    for item in tqdm(data_path):
        img = Image.open(item).convert('L')
        l_img = (np.array(img.resize((128,128))).astype(np.float32)/255.0).flatten()
        l_mean.append(np.mean(l_img))
    
    l_mean = np.mean(l_mean)

    for item in tqdm(data_path):
        img = Image.open(item).convert('L')
        l_img = (np.array(img.resize((128,128))).astype(np.float32)/255.0).flatten()
        l_std.append(np.mean(np.power(l_img - l_mean,2)))

    l_std = np.sqrt(np.mean(l_std))

    print('l mean:%.3f' % l_mean)
    print('l std:%.3f' % l_std)


def voice_time(input_path,save_path):
    info = []
    for subitem in os.scandir(input_path):
        if subitem.is_dir():
            for item in os.scandir(subitem.path):
                info_item = [item.name]
                try:
                    info_item.append(librosa.get_duration(filename=item.path))
                except:
                    info_item.append(-1)
                info.append(info_item)
        else:
            info_item = [subitem.name]
            try:
                info_item.append(librosa.get_duration(filename=subitem.path))
            except:
                info_item.append(-1)
            info.append(info_item)
    
    col = ['file','time']
    csv_file = pd.DataFrame(columns=col,data=info)
    csv_file.to_csv(save_path,index=False)
    

if __name__ == '__main__':

    input_csv = './csv_file/MLC_gamma1mm.csv'
    path_list = pd.read_csv(input_csv)['id'].values.tolist()
    test_input_csv = './csv_file/MLC_gamma1mm_test.csv'
    test_path_list = pd.read_csv(test_input_csv)['id'].values.tolist()
    path_list += test_path_list
    # cal_mean_std(path_list)
    cal_mean_std_single(path_list)
