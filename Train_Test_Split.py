import random
lst = [[x for x in range(30)] for y in range(30)]
for i in lst:
    random.shuffle(i)
    
f = open("test.txt", "w")
for i in range(len(lst)):
    for j in range(5):
        f.write('data/ts/image' + str(i) + '-' + str(lst[i][j]).zfill(2) + '.jpg\n')
f.close()

f = open("train.txt", "w")
for i in range(len(lst)):
    for j in range(5, 30):
        f.write('data/ts/image' + str(i) + '-' + str(lst[i][j]).zfill(2) + '.jpg\n')
f.close()