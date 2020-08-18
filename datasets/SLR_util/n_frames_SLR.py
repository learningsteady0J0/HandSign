from __future__ import print_function, division
import os
import sys
import subprocess

def class_process(dir_path, class_name):
  class_path = os.path.join(dir_path, class_name)
  if not os.path.isdir(class_path):
    return

  for file_name in os.listdir(class_path): 
    video_dir_path = os.path.join(class_path, file_name) # 해당 클래스에서 비디오 폴더 pth    KSL/hi/01 , KSL/hi/02 ...
    image_indices = []
    for image_file_name in os.listdir(video_dir_path): #영상 폴더 안에 있는 이미지들에 접근.
      if 'image' not in image_file_name: # 이미지 파일이름에 image가 없다면
        continue
      image_indices.append(int(image_file_name[6:11])) # 이미지 파일 이름예시 : image_00021.jpg => 6:11이면 00021만 반환.

    if len(image_indices) == 0:
      print('no image files', video_dir_path)
      n_frames = 0
    else:
      image_indices.sort(reverse=True) # 반대로 정렬
      n_frames = image_indices[0] # 반대로 정렬했으니 0이 마지막 프레임.
      print(video_dir_path, n_frames)
    with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file: # n_frames을 str형태로 해당 클래스에서 비디오 폴더 안에 즉 이미지 파일들과 함께 n_frames란 파일이 생성됨.
      dst_file.write(str(n_frames+1))


if __name__=="__main__":
  dir_path = sys.argv[1] # 데이터셋 폴더 경로
  for class_name in os.listdir(dir_path): # 해당 데이터셋 안에 있는 class들을 하나 하나 접근. KSL/hi , KSL/good ...
    class_process(dir_path, class_name)
