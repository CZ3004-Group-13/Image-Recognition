import os
import random

lst = []
prev = "0"
for subdir, dirs, files in os.walk(".\\ts"):
    tempList = []
    for file in files:
        if (file.endswith('.jpg')):
            if (file.startswith('image' + prev + '-')):
                tempList.append('data/ts/' + file)
            else:
                lst.append(tempList)
                prev = file.split('e')[1].split('-')[0]
                print(prev)
                tempList = ['data/ts/' + file]
    lst.append(tempList)
               
for i in lst:
    random.shuffle(i)
    
f = open("test.txt", "w")
for i in range(len(lst)):
    for j in range((int) (len(lst[i]) / 5)):
        f.write(lst[i][j] + '\n')
f.close()

f = open("train.txt", "w")
for i in range(len(lst)):
    for j in range((int) (len(lst[i]) / 5), len(lst[i])):
        f.write(lst[i][j] + '\n')
f.close()
