import os
import socket

import cv2
import numpy as np

import darknet

PORT = 3055
FORMAT = 'utf-8'
SERVER = '192.168.13.13'
ADDRESS = (SERVER, PORT)

ir_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ir_socket.connect(ADDRESS)

WEIGHT_FILE_PATH = './backup/yolov4_custom_train_final.weights'
CONFIG_FILE_PATH = './cfg/yolov4_custom_test.cfg'
DATA_FILE_PATH = './data/yolov4.data'
RPI_IP = '192.168.13.13'
MJPEG_STREAM_URL = 'http://' + RPI_IP + '/html/cam_pic_new.php'
YOLO_BATCH_SIZE = 4
THRESH = 0.9

# change this directory accordingly
os.chdir("C:\\darknet\\darknet-master\\build\\darknet\\x64")


def stack_images(scale, img_array):
    """
    Given array of arrays, stack images into one giant image.
    """

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
    """
    Captures a frame from mjpeg stream and returns an OpenCV image.
    """

    cap = cv2.VideoCapture(MJPEG_STREAM_URL)
    ret, frame = cap.read()
    return frame


def image_detection(image, network, class_names, class_colors, thresh):
    """
    Darknet doesn't accept numpy images.
    Create one with image we reuse for each detect.
    Modified from darknet_images.py.
    Takes in direct image instead of path.
    """

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


def chunks(lst, n):
    """
    Splits an array of images into chunks of size n.
    """

    for counter in range(0, len(lst), n):
        yield lst[counter:counter + n]


def show_all_images(frame_list):
    """
    Shows all the images in an array.
    """

    blank_image = np.zeros_like(frame_list[0])
    for i in range(3):
        if len(frame_list) % 3 == 0:
            break
        frame_list.append(blank_image)

    formatted_list = tuple(chunks(frame_list, 3))

    img_stack = stack_images(1, formatted_list)
    cv2.imshow("Images", img_stack)
    cv2.imwrite("detected_images.jpg", img_stack)

    # Write into one more location because I am kiasu
    cv2.imwrite("test/detected_images.jpg", img_stack)

    if cv2.waitKey() & 0xFF == ord('q'):
        cv2.destroyAllWindows()


def continuous_detect():
    """
    Runs detection continuously.
    """

    mapping = {
        "bullseye": 0,
        "up":       1,
        "down":     2,
        "right":    3,
        "left":     4,
        "stop":     5,
        "one":      6,
        "two":      7,
        "three":    8,
        "four":     9,
        "five":     10,
        "six":      11,
        "seven":    12,
        "eight":    13,
        "nine":     14,
        "A":        15,
        "B":        16,
        "C":        17,
        "D":        18,
        "E":        19,
        "F":        20,
        "G":        21,
        "H":        22,
        "S":        23,
        "T":        24,
        "U":        25,
        "V":        26,
        "W":        27,
        "X":        28,
        "Y":        29,
        "Z":        30
    }

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
        to_stop = False

        imagecounter = 0
        counter = 5

        while not to_stop or counter > 0:
            cv2.waitKey(50)

            msg = read()

            print(msg)
            if msg[0] == "R":  # take photo command
                pass
            elif msg[0] == "S":  # stop img rec after last send
                to_stop = True
            else:
                continue

            obstacle_id = msg[1:]

            img_rec_string = "Nothing detected..."

            # structure: in a list, (id, confidence, [(bbox)])
            # index: 0-id 1-confidence 2-bbox
            # bbox: x,y,w,h

            frame = retrieve_img()
            image, detections = image_detection(frame, network, class_names, class_colors, THRESH)

            # keep track of bigger image
            curr_height = 0

            for i in detections:
                image_id = i[0]  # string
                bbox = i[2]  # tuple
                x_coordinate = int(bbox[0])
                height = int(bbox[3])

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

                if x_coordinate < 83:
                    position = "LEFT"
                elif x_coordinate < 166:
                    position = "CENTRE"
                else:
                    position = "RIGHT"
                    # 250 is maximum

                # find the bigger image and don't detect bullseye
                if height > curr_height and image_id != "bullseye":
                    curr_height = height

                    img_rec_string = obstacle_id + "|" + str(mapping[image_id]) + "|" + str(
                        distance) + "|" + position

                    results[obstacle_id] = i
                    images[obstacle_id] = image

            if obstacle_id not in images:
                ir_socket.send(img_rec_string.encode(FORMAT))
                if to_stop:
                    counter -= 1
                continue

            if to_stop:
                counter = 0

            # draw text of image
            images[obstacle_id] = cv2.putText(images[obstacle_id],
                                              "Obstacle #" + obstacle_id + ": " + results[obstacle_id][0] + "(" + str(
                                                  mapping[results[obstacle_id][0]]) + ")", (10, 20),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            cv2.imwrite("images" + str(imagecounter) + ".jpg", images[obstacle_id])
            cv2.imwrite("test/images" + str(imagecounter) + ".jpg", images[obstacle_id])

            imagecounter += 1

            message = img_rec_string.encode(FORMAT)
            ir_socket.send(message)

    except KeyboardInterrupt:
        print('End of image recognition.')

    # generate image mosaic
    result_frame_list = list(images.values())
    show_all_images(result_frame_list)

def read():
    """
    Reads message from server.
    """
    try:
        msg = ir_socket.recv(1024).decode()
        return msg
    except socket.error as e:
        print("exception: ", e)


if __name__ == "__main__":
    continuous_detect()
