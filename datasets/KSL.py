import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import random
from numpy.random import randint

from utils import load_value_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader): # 해당 영상 폴더에 접근하여 각 이미지들을 읽어 온다.
    video = [] 
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:a
            return video

    return video # 각 이미지의 경로가 저장되어 있는 리스트를 반환.


def get_default_video_loader():
    image_loader = get_default_image_loader() # PIL 사용이냐 accimage 사용이냐 결정.
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']: # 각 클래스 라벨들에 접근. 
        class_labels_map[class_label] = index 
        index += 1
    return class_labels_map # { 'a' : 0 , 'b' : 1 , ' c' : 2 ...} # 그냥 각 라벨에 번호를 지정해주는 것. 


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items(): # key : 영상 이름 , value : { 'subset' : ~~ , 'annotations' : { 'label' : 해당 영상 라벨 } }
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))  # label은 해당 클래스 라벨이고 key는 영상이름임.
            annotations.append(value['annotations']) # {'label' :해당 되는 클래스 라벨}  : 이걸 왜 또 추가했지??

    return video_names, annotations # 둘다 리스트.


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path) # json 파일 load
    video_names, annotations = get_video_names_and_annotations(data, subset) # video_names = [ 'label/'key(영상이름)', 'label/key' ... ], annotations = [ value['annotations'], ...  ]
    class_to_idx = get_class_labels(data) # class_to_idx = { 'a' : 0 , 'b' : 1 , 'c' : 2 ...}
    idx_to_class = {}
    for name, label in class_to_idx.items():
        
        idx_to_class[label] = name  # idx_to_class = { 0 : 'a' , 1 : 'b' , 2 : 'c' ...}

    dataset = [] # 각 비디오 영상에 대한 샘플들 저장.
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i]) # root_path/label/videoname 그냥 폴더를 뜻함.
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames') # n_frames 폴더에 접근 n_frames는 n_frames_ucf101_hmdb51.py로 생성.
        n_frames = int(load_value_file(n_frames_file_path)) # n_frames 는 영상 폴더 안에 있는 이미지 수.
        if n_frames <= 0:
            continue

        begin_t = 1 # 이미지의 시작
        end_t = n_frames # 이미지의 끝
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1] # 영상 이름.
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']] # calss_to_idx(해당 라벨값) => 해당 라벨의 인덱스 값 반환. hi 라벨의 인덱스는 0이면 0반환.
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1: # 이것만 생각 하기.
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class KSL(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index): # index 접근을 위한 것.  a[1] 요런거.
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video'] # 비디오 폴더의 경로.  해당 폴더 안에는 이미지들의 모음집이 있다.

        frame_indices = self.data[index]['frame_indices'] # 총 프레임 개수 
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)  # 프레임들을 가지고 장난치기 -> 그리니깐 temporal(시간)임/
        clip = self.loader(path, frame_indices) # clip = 해당 영상에서의 이미지들 경로의 리스트 값
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip] # 클립에 포함되어 있는 각 이미지에 transform 적용. (random crop 등)
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3) # 하나의 클립에 각 이미지 파일들을 이어 붙임. permute는 축을 이동시키는건데, 직접 찍어봐야지 감이 올듯?
        # torch.stack 은 concatenate(이어붙이기)를 해주는 것. tensor a = [1,4] b= [2,5] c= [6,3] 이 있따면
        # torch.stack([a,b,c]) = tensor [ [1., 4.] , [2., 5.], [6., 3.] ] 요래 됨.

        target = self.data[index] # 해당 영상 정보들이 담겨있는 샘플.
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target # 이미지들의 경로가 있는 클립과  해당 동영상의 정보가 있는 target이 반환.

    def __len__(self):
        return len(self.data)
