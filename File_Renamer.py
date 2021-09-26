import os

j = 0
for i in range(0, 1546):
    if (os.path.isfile(str(i) + '.jpg')):
        os.rename(str(i) + '.jpg',str(j) + '.jpg')
        os.rename(str(i) + '.txt',str(j) + '.txt')
        j += 1