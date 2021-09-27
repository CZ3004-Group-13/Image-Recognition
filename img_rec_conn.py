import darknet
import cv2
import numpy as np
import imutils
import random
import time
import os

#Setup sending of string and receiving of coordinate
import socket
import threading
PORT = 3053
FORMAT = 'utf-8'
SERVER = '192.168.13.13'
ADDR = (SERVER, PORT)

#robot_coord = 'empty'

ir_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ir_socket.connect(ADDR)


WEIGHT_FILE_PATH = './backup/yolov4_custom_train_final.weights'
CONFIG_FILE_PATH = './cfg/yolov4_custom_test.cfg'
DATA_FILE_PATH = './data/yolov4.data'
RPI_IP = '192.168.13.13'
MJPEG_STREAM_URL = 'http://' + RPI_IP + '/html/cam_pic_new.php'
YOLO_BATCH_SIZE = 4
THRESH = 0.85 #may want to lower and do filtering for specific images later

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def retrieve_img():
    #captures a frame from mjpeg stream
    #returns opencv image
    cap = cv2.VideoCapture(MJPEG_STREAM_URL)
    ret, frame = cap.read()
    return frame

def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    #Modified from darknet_images.py
    #Takes in direct image instead of path
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def show_all_images(frame_list):
    #for index, frame in enumerate(frame_list):
    #    frame = imutils.resize(frame, width=400)
    #    cv2.imshow('Image' + str(index), frame)
        
    imgStack = stackImages(2, frame_list)
    cv2.imshow("Images", imgStack)
    cv2.imwrite("test/detected.jpg", imgStack)

    if cv2.waitKey() & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def test_detect():
    frame = cv2.imread('C:\\darknet\\darknet-master\\build\\darknet\\x64\\examples\\test_image_1.jpg')
    #frame = retrieve_img()
    network, class_names, class_colors = darknet.load_network(
        CONFIG_FILE_PATH,
        DATA_FILE_PATH,
        WEIGHT_FILE_PATH,
        YOLO_BATCH_SIZE
    )

    image, detections = image_detection(frame, network, class_names, class_colors, THRESH)
    print(detections)
    cv2.imshow('Inference', image)
    if cv2.waitKey() & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    cv2.imwrite('./result.jpeg', image)

def continuous_detect():
    #use dictionary to store results
    #structure: dictionary, tuple of (id, confidence,(bbox))
    #bbox: x,y,w,h
    #global robot_coord
    #local_robot_coord = 'empty'
    #if robot_coord != 'empty':
    #    local_robot_coord = robot_coord
    #    robot_coord = 'empty'
    
    #local_robot_coord = '(1,1)|N'

    results = {}
    images = {}
    network, class_names, class_colors = darknet.load_network(
        CONFIG_FILE_PATH,
        DATA_FILE_PATH,
        WEIGHT_FILE_PATH,
        YOLO_BATCH_SIZE
    )
    try:
        print('Image recognition started!')
        while True:
            #print('Robot coordinates: ' + local_robot_coord)
            cv2.waitKey(50)
            frame = retrieve_img()
            image, detections = image_detection(frame, network, class_names, class_colors, THRESH)
            
            #structure: in a list, (id, confidence, (bbox))
            #[('9', '99.72', (377.555419921875, 147.49517822265625, 87.70740509033203, 173.86444091796875)), ('7', '99.95', (43.562461853027344, 134.47283935546875, 91.14225006103516, 181.6890411376953)), ('8', '99.96', (214.2314453125, 143.147216796875, 85.68460845947266, 166.68231201171875))]
            #index: 0-id 1-confidence 2-bbox
            #bbox: x,y,w,h
            for i in detections:
                id = i[0] #string
                confidence = i[1] #string
                bbox = i[2] #tuple
                x_coordinate = int(bbox[0])
                y_coordinate = int(bbox[1])
                width = int(bbox[2])
                height = int(bbox[3])
                print('ID detected: ' + id, ', confidence: ' + confidence + ', bbox:' + '[(' + str(x_coordinate) + ', ' + str(y_coordinate) + '), ' + str(width) + ', ' + str(height) + ']')
                if id in results:
                    # print('ID has been detected before') # USEFUL 
                    if float(confidence) > float(results[id][1]):
                        # print('Confidence higher. Replacing existing image.') # USEFUL 
                        del results[id] #remove existing result from dict
                        del images[id] #remove existing img from dict
                        results[id] = i #add new result to dict. DUPLICATE ID IN VALUE PAIR!
                        images[id] = image #add new result to dict
                    else:
                        # print('Confidence lower. Keeping existing image.') # USEFUL
                        pass
                else:
                    # print('New ID. Saving to results and image dict.') # USEFUL
                    results[id] = i
                    images[id] = image
    except KeyboardInterrupt:
        print('End of image recognition.')
    
    #generate string
    # img_rec_result_string = '{'
    img_rec_result_string = ''
    print("Detection results:")
    
    for i in results:
        x_coordinate = int(results[i][2][0])
        y_coordinate = int(results[i][2][1])
        width = int(results[i][2][2])
        height = int(results[i][2][3])
        id_coordinate_str = '[' + i + ', (' + str(x_coordinate) + ', ' + str(y_coordinate) + '), ' + str(width) + ', ' + str(height) + ']' + os.linesep
        img_rec_result_string += id_coordinate_str
        
        # send string to rpi
        # message = img_rec_result_string.encode(FORMAT)
        # ir_socket.send(message)
        time.sleep(0.1)
        #finish send string to rpi

        print('ID: ' + i + ', Coordinates: (' + str(x_coordinate) +',' + str(y_coordinate) + ')' + ', Confidence: ' + results[i][1] + ', bbox:', results[i][2])

    #if img_rec_result_string[-1] == ',':
    #    img_rec_result_string = img_rec_result_string[:-1]
    # img_rec_result_string += '}'
    print(img_rec_result_string)
    message = img_rec_result_string.encode(FORMAT)
    ir_socket.send(message)

    #generate image mosaic
    result_frame_list = list(images.values())
    show_all_images(result_frame_list)

def readRPI():
    while True:
        msg = ir_socket.recv(1024)
        if msg:
            print('Received coordinates')
            robot_coord = msg

            

if __name__ == "__main__":
    #test_detect()
    #read_rpi_thread = threading.Thread(target = readRPI, name = "read_rpi_thread")
    #read_rpi_thread.daemon = True
    #print('Starting RPi comm thread...')
    #read_rpi_thread.start()
    #print('RPi comm thread started.')
    continuous_detect()