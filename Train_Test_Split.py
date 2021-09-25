f = open("test.txt", "w")
for i in range(0, 1546, 2):
    f.write("data/ts/" + str(i) + ".jpg\n")
f.close()

f = open("train.txt", "w")
for i in range(1, 1546, 2):
    f.write("data/ts/" + str(i) + ".jpg\n")
f.close()
