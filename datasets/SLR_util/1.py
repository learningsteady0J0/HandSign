import os
import sys
import shutil
import random

#42:8
dir_path = '/workspace/3D/pytorch/HandSign/SLR/'
train_file = open("trainlist01.txt", 'w')
test_file = open("testlist01.txt", 'w')
numlist = [x for x in range(1,51)]

for class_name in os.listdir(dir_path): # 해당 데이터셋 안에 있는 class들을 하나 하나 접근. KSL/hi , KSL/good ...
    num = 1
    test_file_list = random.sample(numlist,8)
    class_path = os.path.join(dir_path, class_name)

    for file_name in os.listdir(class_path):
        train_file.write("{}/{}\n".format(class_name,file_name))
        if num in test_file_list:
            test_file.write("{}/{}\n".format(class_name,file_name))
        num+=1


'''
#making classInd
dir_path = '/workspace/3D/pytorch/HandSign/SLR/'
str_list = []
for class_name in os.listdir(dir_path): # 해당 데이터셋 안에 있는 class들을 하나 하나 접근. KSL/hi , KSL/good ...
    str_list.append(int(class_name))
print(len(str_list))
str_list = sorted(str_list)

f = open("classInd.txt", 'w')
for i in str_list:
	f.write("{0} {1:06d}\n".format(i,i))
'''