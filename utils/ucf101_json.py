from __future__ import print_function, division
import os
import sys
import json
import pandas as pd

def convert_csv_to_dict(csv_path, subset): # 각 영상에 대한 정보를 데이터베이스 화 하는 함수. 데이터 베이스는 딕셔너리 임.
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    keys = []  # 영상들의 이름 리스트
    key_labels = []  # 영상의 클래스 리스트
    for i in range(data.shape[0]):
        row = data.ix[i, :]
        slash_rows = data.ix[i, 0].split('/') # 파일을 살펴보면 클래스 이름/영상이름.avi로 되어있음 이걸 '/'을 기준으로 split'
        class_name = slash_rows[0] # 클래스 이름
        basename = slash_rows[1].split('.')[0] # A.avi면 A가 basename.
        
        keys.append(basename)
        key_labels.append(class_name)
        
    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset # 'training' or 'validation'
        label = key_labels[i]
        database[key]['annotations'] = {'label': label}
    
    return database

def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.ix[i, 1])
    return labels

def convert_ucf101_csv_to_activitynet_json(label_csv_path, train_csv_path, 
                                           val_csv_path, dst_json_path):
    labels = load_labels(label_csv_path) # 라벨에 대한 리스트 생성
    train_database = convert_csv_to_dict(train_csv_path, 'training')
    val_database = convert_csv_to_dict(val_csv_path, 'validation')
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

if __name__ == '__main__':
    csv_dir_path = sys.argv[1]

    for split_index in range(1, 4):
        label_csv_path = os.path.join(csv_dir_path, 'classInd.txt')
        train_csv_path = os.path.join(csv_dir_path, 'trainlist0{}.txt'.format(split_index))
        val_csv_path = os.path.join(csv_dir_path, 'testlist0{}.txt'.format(split_index))
        dst_json_path = os.path.join(csv_dir_path, 'ucf101_0{}.json'.format(split_index))

        convert_ucf101_csv_to_activitynet_json(label_csv_path, train_csv_path,
                                               val_csv_path, dst_json_path)
