import random
import math


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst


class ClassLabel(object):

    def __call__(self, target): # data load에서 받아온 값에서 'label'값 반환 ( label은 인덱스화 되어 있다.)
        return target['label']


class VideoID(object):

    def __call__(self, target):
        return target['video_id'] # video_id는 해당 영상의 이름, hi 폴더 안에 있는 hi_01 영상이 있으면 hi_01
