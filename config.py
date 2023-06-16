 
__all__ = ['resnet18','resnet34', 'resnet50','resnest18','resnest50',\
            'efficientnet-b5','efficientnet-b7','efficientnet-b3','efficientnet-b8','densenet121',\
            'densenet169','se_resnet18', 'se_resnet50','regnetx-200MF','regnety-8.0GF',\
            'regnety-16GF','res2next50','regnetx-600MF','regnety-600MF','regnety-4.0GF',\
            'hybridnet_v1','swin_transformer','vit_12x12','hybridnet_v2','hybridnet_v3',\
            'hybridnet_v4','hybridnet_v5','mpnet_12x32','hybridnet_v6','mpnet_swin_12x32']


data_config = {
    'MLC_v2':'./converter/csv_file/MLC_v2.csv',
    'MLC':'./converter/csv_file/MLC.csv',
    'MLC_Dose':'./converter/csv_file/MLC_dose_v2.csv',
    'MLC_Gamma1mm':'./converter/csv_file/MLC_gamma1mm_v2.csv',
    'MLC_Gamma2mm':'./converter/csv_file/MLC_gamma2mm_v2.csv',
}

num_classes = {
    'MLC_v2':5,
    'MLC':5,
    'MLC_Dose':5,
    'MLC_Gamma1mm':5,
    'MLC_Gamma2mm':5
}

TASK = 'MLC_Dose'
NET_NAME = 'hybridnet_v5' #regnetx-200MF
VERSION = 'v27.0-x3' 
DEVICE = '1'
# Must be True when pre-training and inference
PRE_TRAINED = True	
# 1,2,3,4,5
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5
TTA_TIMES = 11



NUM_CLASSES = num_classes[TASK]
from utils import get_weight_path,get_weight_list

CSV_PATH = data_config[TASK]
CKPT_PATH = './ckpt/{}/{}/fold{}'.format(TASK,VERSION,str(CURRENT_FOLD))
# CKPT_PATH = './ckpt/{}/{}/fold{}'.format(TASK,'v21.0',str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

if PRE_TRAINED:
    # WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/{}/'.format(TASK,'v21.0'))
    WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/{}/'.format(TASK,VERSION))
else:
    WEIGHT_PATH_LIST = None

# WEIGHT_PATH_LIST =
# 
#  None

MEAN = {
    'MLC_v2':[0.493,0.495,0.501],
    'MLC':[0.499, 0.492, 0.493],
    'MLC_Dose':[0.501],
    'MLC_Gamma1mm':[0.493],
    'MLC_Gamma2mm':[0.495]
}

STD = {
    'MLC_v2':[0.147,0.113,0.062],
    'MLC':[0.065, 0.152, 0.118],
    'MLC_Dose':[0.062],
    'MLC_Gamma1mm':[0.147],
    'MLC_Gamma2mm':[0.113]
}

MILESTONES = {
    'MLC_v2':[30,60,90],
    'MLC':[30,60,90],
    'MLC_Dose':[30,60,90],
    'MLC_Gamma1mm':[30,60,90],
    'MLC_Gamma2mm':[30,60,90]
}

EPOCH = {
    'MLC_v2':150,
    'MLC':150,
    'MLC_Dose':150,
    'MLC_Gamma1mm':150,
    'MLC_Gamma2mm':150
}

TRANSFORM = {
    'MLC_v2':[2,3,4,6,7,8,9,19] if 'norm' not in VERSION else [2,3,4,6,7,8,9,10,19],
    'MLC':[2,3,4,6,7,8,9,19] if 'norm' not in VERSION else [2,3,4,6,7,8,9,10,19], 
    'MLC_Dose':[2,3,4,6,7,8,9,19],
    'MLC_Gamma1mm':[2,3,4,6,7,8,9,19],
    'MLC_Gamma2mm':[2,3,4,6,7,8,9,19]
}
print('transform list:',TRANSFORM[TASK])
SHAPE = {
    'MLC_v2':(224,224),
    'MLC':(224, 224),
    'MLC_Dose':(224, 224),
    'MLC_Gamma1mm':(224, 224),
    'MLC_Gamma2mm':(224, 224)
}


CHANNEL = {
    'MLC_v2':3,
    'MLC':3,
    'MLC_Dose':1,
    'MLC_Gamma1mm':1,
    'MLC_Gamma2mm':1
}

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-5 if not PRE_TRAINED else 5e-4, #1e-3
    'n_epoch': EPOCH[TASK],
    'channels': CHANNEL[TASK],
    'num_classes': NUM_CLASSES,
    'input_shape': SHAPE[TASK],
    'crop': 0,
    'batch_size': 32, #32
    'num_workers': 2,
    'device': DEVICE,
    'pre_trained': PRE_TRAINED,
    'weight_path': WEIGHT_PATH,
    'weight_decay': 0.0001, #0.0001
    'momentum': 0.9,
    'mean': MEAN[TASK],
    'std': STD[TASK],
    'gamma': 0.1,
    'milestones': MILESTONES[TASK],
    'use_fp16':True,
    'transform':TRANSFORM[TASK],
    'drop_rate': 0.5, #0.5
    'smoothing':0.15,
    'external_pretrained':True if 'pretrained' in VERSION else False,#False
    'use_mixup':True if 'mixup' in VERSION else False,
    'use_cutmix':True if 'cutmix' in VERSION else False,
    'mix_only': True if 'only' in VERSION else False,
    'use_maxpool': False or 'maxpool' in VERSION
}


# Arguments when perform the trainer
__loss__ = ['Cross_Entropy','TopkCrossEntropy','SoftCrossEntropy','F1_Loss','TopkSoftCrossEntropy','DynamicTopkCrossEntropy','DynamicTopkSoftCrossEntropy']
LOSS_FUN = 'Cross_Entropy'
# Arguments when perform the trainer

print('loss fun:%s'%LOSS_FUN)
print('network:%s'%NET_NAME)

CLASS_WEIGHT = {
    'MLC_v2':None,
    'MLC':None,
    'MLC_Dose':None,
    'MLC_Gamma1mm':None,
    'MLC_Gamma2mm':None
}


MONITOR = {
    'MLC_v2':'val_f1',
    'MLC':'val_f1',
    'MLC_Dose':'val_f1',
    'MLC_Gamma1mm':'val_f1',
    'MLC_Gamma2mm':'val_f1'
}

SETUP_TRAINER = {
    'output_dir': './ckpt/{}/{}'.format(TASK,VERSION),
    'log_dir': './log/{}/{}'.format(TASK,VERSION),
    'optimizer': 'AdamW', 
    'loss_fun': LOSS_FUN,
    'class_weight': CLASS_WEIGHT[TASK],
    'lr_scheduler': 'CosineAnnealingWarmUp', #'MultiStepLR','CosineAnnealingWarmRestarts', 'CosineAnnealingWarmUp' for fine-tune and warmup
    'balance_sample':True if 'balance' in VERSION else False,#False
    'monitor':MONITOR[TASK],
    'repeat_factor':3.0,
}
