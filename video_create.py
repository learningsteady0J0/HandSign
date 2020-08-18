import cv2
import os

'''
for i in range(1,51):
	label = i
	pathIn= './20bn-jester-v1/{}/'.format(label)
	pathOut = './video/{}.avi'.format(label)
	fps = 25
	frame_array = []
	paths = sorted(os.listdir(pathIn))
	del paths[-1]

	for idx , path in enumerate(paths) : 
	    #if (idx % 2 == 0) | (idx % 5 == 0) :
	    #    continue
	    path = os.path.join(pathIn, path)
	    img = cv2.imread(path)
	    height, width, layers = img.shape
	    size = (width,height)
	    frame_array.append(img)
	    print(size)

	out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MPEG'), fps, size)
	for i in range(len(frame_array)):
	    # writing to a image array
	    out.write(frame_array[i])
	out.release()
'''

label = 111255
pathIn= './20bn-jester-v1/{}/'.format(label)
pathOut = './video/{}Pushing_Two_Fingers_Away.avi'.format(label)
fps = 20
frame_array = []
paths = sorted(os.listdir(pathIn))
del paths[-1]

for idx , path in enumerate(paths) : 
    #if (idx % 2 == 0) | (idx % 5 == 0) :
    #    continue
    path = os.path.join(pathIn, path)
    img = cv2.imread(path)
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
    print(size)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MPEG'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()
