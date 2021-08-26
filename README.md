# MDP Image Recognition
This repository contains several sections.

## Google Colab
`Image_Recognition.ipynb` contains code to run darknet on Google Colab.

## Helpful Files
`Photo_Taking_Helper.py` & `Train_Test_Split.ipynb` are helpful files to help prepare data.

## DarkNet on Computer
If you wish to run DarkNet on your computer, here are some useful links and commmands:

- [OpenCV](https://www.youtube.com/watch?v=YsmhKar8oOc)
- [DarkNet](https://www.youtube.com/watch?v=FE2GBeKuqpc)

### Image:
- `darknet.exe detector test data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights`
- `darknet.exe detector test data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_image_1.jpg`
- `darknet.exe detector test data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_image_2.jpg`
- `darknet.exe detector test data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_image_3.jpg`
- `darknet.exe detector test data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_image_4.jpg`

### Video:
- `darknet.exe detector demo data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_video.mp4`
- `darknet.exe detector demo data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights examples/test_video.mp4 -out-filename test/output.mp4`

### Webcam:
- `darknet.exe detector demo data/yolov4.data cfg/yolov4_custom_test.cfg backup/yolov4_custom_train_last.weights`
