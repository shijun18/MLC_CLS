 
__all__ = ['resnet18','resnet34', 'resnet50','resnest18','resnest50',\
            'efficientnet-b5','efficientnet-b7','efficientnet-b3','efficientnet-b8','densenet121',\
            'densenet169','regnetx-200MF','regnetx-600MF','regnety-600MF','regnety-4.0GF',\
            'regnety-8.0GF','regnety-16GF','res2next50', 'simplenet','simplenetv2',\
            'simplenetv3','simplenetv4','simplenetv5','simplenetv6','simplenetv7',\
            'simplenetv8','simplenetv9','simplenetv10','res2next18','se_resnet18', 'se_resnet10',\
            'bilinearnet_b5','finenet50','directnet50']



data_config = {
    'MLC':'./converter/csv_file/MLC.csv',
    'MLC_Dose':'./converter/csv_file/MLC_dose.csv',
    'MLC_Gamma1mm':'./converter/csv_file/MLC_gamma1mm.csv',
    'MLC_Gamma2mm':'./converter/csv_file/MLC_gamma2mm.csv',
}

num_classes = {
    'MLC':5,
    'MLC_Dose':5,
    'MLC_Gamma1mm':5,
    'MLC_Gamma2mm':5
}

TASK = 'MLC'
NET_NAME = 'efficientnet-b5' #regnetx-200MF
VERSION = 'v6.0-pretrained-x1' 
DEVICE = '0,1'
# Must be True when pre-training and inference
PRE_TRAINED = False	
# 1,2,3,4
CURRENT_FOLD = 5
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

# WEIGHT_PATH_LIST = None

MEAN = {
    'MLC':[0.499, 0.492, 0.493],
    'MLC_Dose':[0.499],
    'MLC_Gamma1mm':[0.492],
    'MLC_Gamma2mm':[0.493]
}

STD = {
    'MLC':[0.065, 0.152, 0.118],
    'MLC_Dose':[0.065],
    'MLC_Gamma1mm':[0.152],
    'MLC_Gamma2mm':[0.118]
}

MILESTONES = {
    'MLC':[30,60,90],
    'MLC_Dose':[30,60,90],
    'MLC_Gamma1mm':[30,60,90],
    'MLC_Gamma2mm':[30,60,90]
}

EPOCH = {
    'MLC':150,
    'MLC_Dose':150,
    'MLC_Gamma1mm':150,
    'MLC_Gamma2mm':150
}

TRANSFORM = {
    'MLC':[2,3,4,6,7,8,9,10,19],
    'MLC_Dose':[2,3,4,6,7,8,9,10,19],
    'MLC_Gamma1mm':[2,3,4,6,7,8,9,10,19],
    'MLC_Gamma2mm':[2,3,4,6,7,8,9,10,19]
}

SHAPE = {
    'MLC':(128, 128),
    'MLC_Dose':(128, 128),
    'MLC_Gamma1mm':(128, 128),
    'MLC_Gamma2mm':(128, 128)
}


CHANNEL = {
    'MLC':3,
    'MLC_Dose':1,
    'MLC_Gamma1mm':1,
    'MLC_Gamma2mm':1
}

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-3 if not PRE_TRAINED else 5e-4, #1e-3
    'n_epoch': EPOCH[TASK],
    'channels': CHANNEL[TASK],
    'num_classes': NUM_CLASSES,
    'input_shape': SHAPE[TASK],
    'crop': 0,
    'batch_size': 64, #32
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
    'smothing':0.15,
    'external_pretrained':True if 'pretrained' in VERSION else False,#False
    'use_mixup':True if 'mixup' in VERSION else False,
    'use_cutmix':True if 'cutmix' in VERSION else False,
    'mix_only': True if 'only' in VERSION else False
}


# Arguments when perform the trainer
__loss__ = ['Cross_Entropy','TopkCrossEntropy','SoftCrossEntropy','F1_Loss','TopkSoftCrossEntropy','DynamicTopkCrossEntropy','DynamicTopkSoftCrossEntropy']
LOSS_FUN = 'Cross_Entropy'
# Arguments when perform the trainer

CLASS_WEIGHT = {
    'MLC':None,
    'MLC_Dose':None,
    'MLC_Gamma1mm':None,
    'MLC_Gamma2mm':None
}


MONITOR = {
    'MLC':'val_acc',
    'MLC_Dose':'val_acc',
    'MLC_Gamma1mm':'val_acc',
    'MLC_Gamma2mm':'val_acc'
}

SETUP_TRAINER = {
    'output_dir': './ckpt/{}/{}'.format(TASK,VERSION),
    'log_dir': './log/{}/{}'.format(TASK,VERSION),
    'optimizer': 'AdamW', 
    'loss_fun': LOSS_FUN,
    'class_weight': CLASS_WEIGHT[TASK],
    'lr_scheduler': 'CosineAnnealingWarmRestarts', #'MultiStepLR','CosineAnnealingWarmRestarts' for fine-tune and warmup
    'balance_sample':True if 'balance' in VERSION else False,#False
    'monitor':MONITOR[TASK],
    'repeat_factor':1.0
}
