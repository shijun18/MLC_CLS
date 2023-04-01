import os 
import pandas as pd 
import glob
import random
import shutil

# INDEX = {
#     'Shift_png':0,
#     'Random_png':1,
#     'Original_png':2,
#     'Expand_png':3,
#     'Contract_png':4
# }

# INDEX = {
#     'Shift':0,
#     'Random':1,
#     'Original':2,
#     'Expand':3,
#     'Contract':4
# }

INDEX = {
    'shift':0,
    'random':1,
    'original':2,
    'expand':3,
    'contract':4
}

def make_label_csv(input_path,csv_path,mid_dir=None):

    info = []
    for subdir in os.scandir(input_path):
        index = INDEX[subdir.name]
        if mid_dir is None:
            path_list = glob.glob(os.path.join(subdir.path,"*.*g"))
        else:
            mid_path = os.path.join(subdir.path,mid_dir)
            # print(mid_path)
            path_list = glob.glob(os.path.join(mid_path,"*.*g"))
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
  

    # input_path = '/staff/shijun/torch_projects/MLC_CLS/dataset/MLC/train'
    # csv_path = './csv_file/MLC.csv'
    # input_path = '/staff/shijun/torch_projects/MLC_CLS/dataset/MLC/test'
    # csv_path = './csv_file/MLC_test.csv'
    # make_label_csv(input_path,csv_path)

    input_path = '/staff/shijun/torch_projects/MLC_CLS/dataset/raw_data_v2/train'
    csv_path = './csv_file/MLC_dose_v2.csv'
    make_label_csv(input_path,csv_path,mid_dir='dose')
    input_path = '/staff/shijun/torch_projects/MLC_CLS/dataset/raw_data_v2/test'
    csv_path = './csv_file/MLC_dose_v2_test.csv'
    make_label_csv(input_path,csv_path,mid_dir='dose')

    # input_path = '/staff/shijun/torch_projects/MLC_CLS/dataset/MLC_v2/train'
    # csv_path = './csv_file/MLC_v2.csv'
    # make_label_csv(input_path,csv_path)
    # input_path = '/staff/shijun/torch_projects/MLC_CLS/dataset/MLC_v2/test'
    # csv_path = './csv_file/MLC_v2_test.csv'
    # make_label_csv(input_path,csv_path)