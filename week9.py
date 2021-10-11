import os
import socket
import threading

import cv2

import darknet

PORT = 3055
FORMAT = "utf-8"
SERVER = "192.168.13.13"
ADDRESS = (SERVER, PORT)

ir_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ir_socket.connect(ADDRESS)

CURRENT_WORKING_PATH = "C:\\darknet\\darknet-master\\build\\darknet\\x64"
WEIGHT_FILE_PATH = "./backup/yolov4_custom_train_final.weights"
CONFIG_FILE_PATH = "./cfg/yolov4_custom_test.cfg"
DATA_FILE_PATH = "./data/yolov4.data"
MJPEG_STREAM_URL = "http://" + SERVER + "/html/cam_pic_new.php"
YOLO_BATCH_SIZE = 4
THRESH = 0.9

# change this directory accordingly
os.chdir(CURRENT_WORKING_PATH)


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


def continuous_detect():
    """
    Runs detection continuously.
    """

    network, class_names, class_colors = darknet.load_network(
        CONFIG_FILE_PATH,
        DATA_FILE_PATH,
        WEIGHT_FILE_PATH,
        YOLO_BATCH_SIZE
    )
    try:
        print("Image recognition started!")
        to_stop = False

        while not to_stop:

            cv2.waitKey(50)

            # msg = read_msg()

            # print(msg)
            # if msg[0] == "R":  # take photo command
            #     pass
            # elif msg[0] == "S":  # stop img rec after last send
            #     to_stop = True
            # else:
            #     continue

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

                if image_id != "bullseye":
                    continue

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
                    # 52 is height of 80 cm

                if x_coordinate < 83:
                    position = "LEFT"
                elif x_coordinate < 166:
                    position = "CENTRE"
                else:
                    position = "RIGHT"
                    # 250 is maximum

                # find the bigger image TODO: this logic might have to change
                if height > curr_height:
                    curr_height = height

                    img_rec_string = str(distance) + "|" + position

            message = img_rec_string.encode(FORMAT)
            ir_socket.send(message)

    except KeyboardInterrupt:
        print("End of image recognition.")


def read_msg():
    """
    Reads message from server.
    """
    try:
        msg = ir_socket.recv(1024).decode()
        return msg
    except socket.error as e:
        print("exception: ", e)


def main():
    """
    Main function
    """

    # read_rpi_thread = threading.Thread(target=read_msg, name="read_rpi_thread")
    # read_rpi_thread.daemon = True
    # print('Starting RPi comm thread...')
    # read_rpi_thread.start()
    # print('RPi comm thread started.')
    continuous_detect()


if __name__ == "__main__":
    main()
