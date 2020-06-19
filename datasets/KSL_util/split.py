import os
import random

base_path = '/workspace/JSW/hand/HandSign'
folder_name = 'KSL-image'

f1 = open('./trainlist01.txt', 'w')
f2 = open('./testlist01.txt', 'w')
f3 = open('./classInd.txt', 'r')

labels = []
key = 0
while True:
    line = f3.readline()
    if not line: break
    if key == 0:
    	labels.append(line[2:-1])
    else:
    	labels.append(line[3:-1])
    if line == '9 bus\n':
    	key = 1

count = 0 
for j in labels:
    test = []
    a = random.randrange(0,19)
    while True:
        b = random.randrange(0,19)
        if a != b:
            test.append(a)
            test.append(b)
            break

    for i in range(20):
	    if not os.path.isdir('{}/{}/{}/{}_{:>02d}/'.format(base_path, folder_name, j, j, i)):
	    	print('{}/{}/{}/{}_{:>02d}'.format(base_path,folder_name, j, j, i))
	    	continue
	    if i in test : #djqtsms dlalwleh dlTrlEoansdp count sms rPthr ekfmrp skdha.
	    	count += 1
	    	f2.write('{}/{}_{:>02d}.MP4\n'.format(j,j,i))
	    else:
	    	f1.write('{}/{}_{:>02d}.MP4\n'.format(j,j,i))

print(count)