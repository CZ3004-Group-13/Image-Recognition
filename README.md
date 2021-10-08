# MDP Image Recognition

This repository contains several sections.

## Google Colab

`Image_Recognition.ipynb` contains code to run darknet on Google Colab.

## Helpful Files

`Photo_Taking_Helper.py`, `File_Renamer.py` & `Train_Test_Split.py` are helpful files to help prepare data.

## DarkNet on Computer

If you wish to run DarkNet on your computer, here are some useful links and commmands:

- [OpenCV](https://www.youtube.com/watch?v=YsmhKar8oOc)
- [DarkNet](https://www.youtube.com/watch?v=FE2GBeKuqpc)
- Navigate to `C:\darknet\darknet-master\build\darknet\x64` on Anaconda Prompt

## RPi-Cam-Web-Interface

Install RPi_Cam_Web_Interface:

1. Make sure you install Raspbian on your RPi
2. Attach camera to RPi
3. Enable camera support - for Desktop Raspbian : sudo raspi-config
4. Update your RPi with the following commands:

- sudo apt-get update
- sudo apt-get dist-upgrade
- sudo apt-get install git

5. Clone the code from github and enable and run the install script with the following commands: git
   clone https://github.com/silvanmelchior/RPi_Cam_Web_Interface.git
   cd RPi_Cam_Web_Interface
6. Lastly, carry on with the installation: ./install.sh

How to use:

- In the terminal window, cd RPi_Cam_Web_Interface
- To start the camera software, use: ./start.sh
- Go to http://192.168.13.13/html/cam_pic_new.php to access the server's video
- To stop the camera software, use: ./stop.sh
- To update an existing installation, use: ./update.sh

## Darknet using RPi-Cam-Web-Interface

1. In command prompt, cd C:\darknet\darknet-master\build\darknet\x64
2. Run following command: darknet.exe detector demo data/yolov4.data cfg/yolov4_custom_test.cfg
   backup/yolov4_custom_train_final.weights http://192.168.13.13/html/cam_pic_new.php

If running aboved command is unsuccessful, install IP Camera Adapter using this link https://ip-webcam.appspot.com/

1. In IP Camera Adapter, set Camera feed URL to : http://192.168.13.13/html/cam_pic_new.php
2. Set video size as 640x480 as darknet only allowed 640x480
3. Once both done, click 'Apply' then 'OK'
4. Lastly repeat the first step in command prompt, then run the following command: darknet.exe detector demo
   data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_final.weights -c 1
5. The real-time image recognition stream with bounding boxes should be running using Darknet and RPi-Cam-Web-Interface

### Image:

- `darknet.exe detector test data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights`
- `darknet.exe detector test data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_image_1.jpg`
- `darknet.exe detector test data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_image_2.jpg`
- `darknet.exe detector test data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_image_3.jpg`
- `darknet.exe detector test data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_image_4.jpg`

### Video:

- `darknet.exe detector demo data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_video.mp4`
- `darknet.exe detector demo data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_video.mp4 -out_filename test/output.mp4`

### Webcam:

- `darknet.exe detector demo data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights`
