from picamera import PiCamera
camera = PiCamera()
id = '0'
os.mkdir(id.zfill(2))
for i in range(20):
    camera.capture(id.zfill(2) + '/image' + id + '-' + str(i).zfill(2) + '.jpg')
    time.sleep(0.5)
	