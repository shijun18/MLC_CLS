import os
import re
import pandas as pd



def statistics_metric(input_path,result_path,net_name,ver=[1.0,2.0,3.0,4.0],class_num=5):
    csv_info = []
    for i in range(1,6):
        for k,j in enumerate(ver):
            item = []
            report_path = f'v{j}/fold{str(i)}_report.csv'
            report_path = os.path.join(input_path,report_path)
            version = f'v{j}-fold{i}'
            print('version:%s,net:%s'%(version,net_name[k]))
            item.append(version)
            item.append(net_name[k])
            print(report_path)
            csv_file = pd.read_csv(report_path,index_col=0)
            print(csv_file)
            csv_file = csv_file.drop(labels='support')  # drop `support`
            csv_file = csv_file.drop(labels=['macro avg','weighted avg'],axis=1)
            print(csv_file)
            accuracy = [csv_file['accuracy'].tolist()[0]]*len(csv_file.columns)
            csv_file.loc['accuracy'] = accuracy
            csv_file = csv_file.drop(labels='accuracy',axis=1)
            csv_file = csv_file.T
            print(csv_file)

            for index in csv_file.index:
                item += csv_file.loc[index].tolist()
            csv_info.append(item)

            print(list(csv_file.columns))
            if i == 5:
                columns = ['version','net_name'] + list(csv_file.columns) * class_num

    csv_file = pd.DataFrame(data=csv_info,columns=columns)
    csv_file.to_csv(result_path,index=False)


def statistics_auc(input_path,result_path,net_name,labels_name,ver=[1.0,2.0,3.0,4.0]):
    from sklearn.metrics import roc_curve,auc
    import numpy as np
    csv_info = []
    for i in range(1,6):
        for k,j in enumerate(ver):
            item = []
            csv_path = f'v{j}/submission_fold{i}.csv'
            csv_path = os.path.join(input_path,csv_path)
            version = f'v{j}-fold{i}'
            print(version)
            item.append(version)
            item.append(net_name[k])

            file_csv = pd.read_csv(csv_path)
            true_ = np.asarray(file_csv['category_id'].values.tolist())
            print(true_.shape)
            prob_list = [f'prob_{str(i+1)}' for i in range(len(labels_name))]
            prob_ = np.asarray(file_csv[prob_list].values)
            # print(prob_.shape)
            for index in range(len(labels_name)):
                fpr,tpr,threshold = roc_curve(y_true=true_,y_score=prob_[:,index],pos_label=index) 
                roc_auc = auc(fpr,tpr)
                item.append(roc_auc)
            
            csv_info.append(item)
            
            if i == 4:
                columns = ['version','net_name'] + labels_name
    
    csv_file = pd.DataFrame(data=csv_info,columns=columns)
    csv_file.to_csv(result_path,index=False)




if __name__ == '__main__':
    
    # input_path = './result/MLC'
    # result_path = './result/result_metric.csv'
    # net_name = ['resnet18','resnet50','resnest50','efficientnet-b5','se_resnet18', 'se_resnet50', \
    #     'hybridnet_v1','swin_transformer','vit_12x12','hybridnet_v2','hybridnet_v3',\
    #     'hybridnet_v3','hybridnet_v3','hybridnet_v4','hybridnet_v4']
    # statistics_metric(input_path,result_path,net_name,['1.0-x1-newaug','3.0-x1-newaug',\
    #     '5.0-x1-newaug','6.0-x1-newaug','12.0-x1-newaug','13.0-x1-newaug','21.0-x1-newaug','22.0-x1-newaug',\
    #     '23.0-x1-newaug','24.0-x1-newaug','25.0-x1-newaug','25.0-x3-newaug','25.0-x6-newaug','26.0-x1-newaug','26.0-x3-newaug',])


    # input_path = './result/MLC'
    # result_path = './result/result_auc.csv'
    # net_name = ['resnet18','resnet50','resnest50','efficientnet-b5','se_resnet18', 'se_resnet50', \
    #     'hybridnet_v1','swin_transformer','vit_12x12','hybridnet_v2','hybridnet_v3']
    # labels_name = [
    #     'Shift_png',
    #     'Random_png',
    #     'Original_png',
    #     'Expand_png',
    #     'Contract_png'
    # ]
    # statistics_auc(input_path,result_path,net_name,labels_name,['1.0-x1-newaug','3.0-x1-newaug',\
    #     '5.0-x1-newaug','6.0-x1-newaug','12.0-x1-newaug','13.0-x1-newaug','21.0-x1-newaug','22.0-x1-newaug',\
    #     '23.0-x1-newaug','24.0-x1-newaug','25.0-x1-newaug'])

    input_path = './result/MLC'
    result_path = './result/result_metric_x1.csv'
    net_name = ['resnet18','resnet50','resnest50','efficientnet-b5','se_resnet18', 'se_resnet50', \
        'hybridnet_v1','swin_transformer','vit_12x12','hybridnet_v3','hybridnet_v3','hybridnet_v3','hybridnet_v4']
    statistics_metric(input_path,result_path,net_name,['1.0-x1','3.0-x1',\
        '5.0-x1','6.0-x1','12.0-x1','13.0-x1','21.0-x3','22.0-x3',\
        '23.0-x3','25.0-x1','25.0-x3','25.0-x6','26.0-x3'])
