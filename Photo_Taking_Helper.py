# Before running, please change to the correct directory!
import os
import time
from picamera import PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
# Change this ID every single time!
id = '0'

for i in range(20):
    camera.capture('image' + id + '-' + str(i).zfill(2) + '.jpg')
    time.sleep(0.5)
print('Done!')
	
