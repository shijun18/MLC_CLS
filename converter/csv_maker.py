import os 
import pandas as pd 
import glob
import random
import shutil

INDEX = {
    'Shift_png':0,
    'Random_png':1,
    'Original_png':2,
    'Expand_png':3,
    'Contract_png':4
}

def make_label_csv(input_path,csv_path):

    info = []
    for subdir in os.scandir(input_path):
        # if subdir.name == '1':
        #     continue
        index = INDEX[subdir.name]
        path_list = glob.glob(os.path.join(subdir.path,"*.*g"))
        sub_info = [[item,index] for item in path_list]
        info.extend(sub_info)
    
    random.shuffle(info)
    # print(len(info))
    col = ['id','label']
    info_data = pd.DataFrame(columns=col,data=info)
    info_data.to_csv(csv_path,index=False)



def make_csv(input_path,csv_path):
    id_list = glob.glob(os.path.join(input_path,'*.*g'))
    print(len(id_list))
    info = {'id':[]}
    info['id'] = id_list
    df = pd.DataFrame(data=info)
    df.to_csv(csv_path,index=False)



if __name__ == "__main__":
  

    input_path = '/staff/shijun/torch_projects/MLC_CLS/dataset/MLC/train'
    csv_path = './csv_file/MLC.csv'
    make_label_csv(input_path,csv_path)