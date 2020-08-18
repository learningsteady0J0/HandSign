import os
import random

dir_path = '/workspace/3D/pytorch/HandSign/KETI_jpg/'
train_file = open("trainlist01.txt", 'w')
test_file = open("testlist01.txt", 'w')
numlist = [x for x in range(1,21)]

for class_name in os.listdir(dir_path): # 해당 데이터셋 안에 있는 class들을 하나 하나 접근. KSL/hi , KSL/good ...
    num = 1
    test_file_list = random.sample(numlist,4)
    class_path = os.path.join(dir_path, class_name)

    for file_name in os.listdir(class_path):
        train_file.write("{}/{}\n".format(class_name,file_name))
        if num in test_file_list:
            test_file.write("{}/{}\n".format(class_name,file_name))
        num+=1