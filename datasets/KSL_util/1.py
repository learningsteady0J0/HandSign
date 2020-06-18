import os

b = []
for foldername in os.listdir('./'):
    b.append(foldername)

b.sort()
b = b[1:]


for foldername in b:
    c = []
    for videoname in os.listdir('./{}'.format(foldername)):
        c.append(videoname) 
        c.sort()
    for d in c:
        os.rename('./{}/{}'.format(foldername, d), './{}/{}_{}'.format(foldername,foldername,d[0:2]))
