import os
# Setup sending of string and receiving of coordinate
import socket
import time

import cv2
import numpy as np

import darknet

PORT = 3053
FORMAT = 'utf-8'
SERVER = '192.168.13.13'
ADDR = (SERVER, PORT)

# robot_coord = 'empty'

ir_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ir_socket.connect(ADDR)

WEIGHT_FILE_PATH = './backup/yolov4_custom_train_final.weights'
CONFIG_FILE_PATH = './cfg/yolov4_custom_test.cfg'
DATA_FILE_PATH = './data/yolov4.data'
RPI_IP = '192.168.13.13'
MJPEG_STREAM_URL = 'http://' + RPI_IP + '/html/cam_pic_new.php'
YOLO_BATCH_SIZE = 4
THRESH = 0.85  # may want to lower and do filter for specific images later


def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def retrieve_img():
    # captures a frame from mjpeg stream
    # returns opencv image
    cap = cv2.VideoCapture(MJPEG_STREAM_URL)
    ret, frame = cap.read()
    return frame


def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    # Modified from darknet_images.py
    # Takes in direct image instead of path
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
    # for index, frame in enumerate(frame_list):
    #    frame = imutils.resize(frame, width=400)
    #    cv2.imshow('Image' + str(index), frame)

    img_stack = stack_images(2, frame_list)
    cv2.imshow("Images", img_stack)
    cv2.imwrite("test/detected.jpg", img_stack)

    if cv2.waitKey() & 0xFF == ord('q'):
        cv2.destroyAllWindows()


# def test_detect():
#     frame = cv2.imread('C:\\darknet\\darknet-master\\build\\darknet\\x64\\examples\\test_image_1.jpg')
#     # frame = retrieve_img()
#     network, class_names, class_colors = darknet.load_network(
#         CONFIG_FILE_PATH,
#         DATA_FILE_PATH,
#         WEIGHT_FILE_PATH,
#         YOLO_BATCH_SIZE
#     )
#
#     image, detections = image_detection(frame, network, class_names, class_colors, THRESH)
#     print(detections)
#     cv2.imshow('Inference', image)
#     if cv2.waitKey() & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#     cv2.imwrite('./result.jpeg', image)


def continuous_detect():
    # use dictionary to store results
    # structure: dictionary, tuple of (id, confidence,(bbox))
    # bbox: x,y,w,h
    # global robot_coord
    # local_robot_coord = 'empty'
    # if robot_coord != 'empty':
    #    local_robot_coord = robot_coord
    #    robot_coord = 'empty'

    # local_robot_coord = '(1,1)|N'

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
            # print('Robot coordinates: ' + local_robot_coord)
            cv2.waitKey(50)
            # UNCOMMENT TO WAIT FOR MESSAGE FROM RPI
            # read_rpi()
            frame = retrieve_img()
            image, detections = image_detection(frame, network, class_names, class_colors, THRESH)

            # structure: in a list, (id, confidence, [(bbox)])
            # index: 0-id 1-confidence 2-bbox
            # bbox: x,y,w,h
            for i in detections:
                image_id = i[0]  # string
                confidence = i[1]  # string
                bbox = i[2]  # tuple
                x_coordinate = int(bbox[0])
                y_coordinate = int(bbox[1])
                width = int(bbox[2])
                height = int(bbox[3])
                print('ID detected: ' + image_id,
                      ', confidence: ' + confidence + ', bbox:' + '[(' + str(x_coordinate) + ', ' + str(
                          y_coordinate) + '), ' + str(width) + ', ' + str(height) + ']')
                if image_id in results:
                    # print('ID has been detected before') # USEFUL 
                    if float(confidence) > float(results[image_id][1]):
                        # print('Confidence higher. Replacing existing image.') # USEFUL 
                        del results[image_id]  # remove existing result from dict
                        del images[image_id]  # remove existing img from dict
                        results[image_id] = i  # add new result to dict. DUPLICATE ID IN VALUE PAIR!
                        images[image_id] = image  # add new result to dict
                    else:
                        # print('Confidence lower. Keeping existing image.') # USEFUL
                        pass
                else:
                    # print('New ID. Saving to results and image dict.') # USEFUL
                    results[image_id] = i
                    images[image_id] = image
    except KeyboardInterrupt:
        print('End of image recognition.')

    # generate string
    # img_rec_result_string = '{'
    img_rec_result_string = ''
    print("Detection results:")

    for i in results:
        x_coordinate = int(results[i][2][0])
        y_coordinate = int(results[i][2][1])
        width = int(results[i][2][2])
        height = int(results[i][2][3])
        id_coordinate_str = '[' + i + ', (' + str(x_coordinate) + ', ' + str(y_coordinate) + '), ' + str(
            width) + ', ' + str(height) + ']' + os.linesep
        img_rec_result_string += id_coordinate_str

        # send string to rpi
        # message = img_rec_result_string.encode(FORMAT)
        # ir_socket.send(message)
        time.sleep(0.1)
        # finish send string to rpi

        if height > 190:
            distance = 15
        elif height > 170:
            distance = 20
        elif height > 150:
            distance = 25
        elif height > 133:
            distance = 30
        elif height > 117:
            distance = 35
        elif height > 95:
            distance = 40
        elif height > 85:
            distance = 45
        elif height > 78:
            distance = 50
        elif height > 72:
            distance = 55
        elif height > 64:
            distance = 60
        elif height > 61:
            distance = 65
        elif height > 58:
            distance = 70
        else:
            distance = 75
        # 52 is height of 80cm

        # 1 LEFT, 2 CENTRE, 3 RIGHT
        if x_coordinate < 83:
            direction = 1
        elif x_coordinate < 166:
            direction = 2
        else:
            direction = 3
        # 250 is maximum

        slant = (width / height < 0.4)
        # if not slant, width / height is approx 0.5

        print('ID: ' + i + ', Coordinates: (' + str(x_coordinate) + ',' + str(y_coordinate) + ')' + ', Confidence: ' +
              results[i][1] + ', bbox:', results[i][2])
        print('Distance:', distance, 'Direction:', direction)

    # if img_rec_result_string[-1] == ',':
    #    img_rec_result_string = img_rec_result_string[:-1]
    # img_rec_result_string += '}'
    print(img_rec_result_string)
    message = img_rec_result_string.encode(FORMAT)
    ir_socket.send(message)

    # generate image mosaic
    result_frame_list = list(images.values())
    show_all_images(result_frame_list)


def read_rpi():
    while True:
        msg = ir_socket.recv(1024)
        if msg:
            print('Received Message from RPi!')
            return msg


if __name__ == "__main__":
    # test_detect()
    # read_rpi_thread = threading.Thread(target = readRPI, name = "read_rpi_thread")
    # read_rpi_thread.daemon = True
    # print('Starting RPi comm thread...')
    # read_rpi_thread.start()
    # print('RPi comm thread started.')
    continuous_detect()
