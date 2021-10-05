import socket
import sys
import traceback
import errno
import time
from config import *


class Algo:
    host = WIFI_IP
    port = WIFI_PORT

    def __init__(self, host=WIFI_IP, port=WIFI_PORT):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("Socket Established")

        try:
            self.socket.bind(('', self.port))
        except socket.error as e:
            print("Bind failed", e)
            sys.exit()

        print("Bind completed")

        self.socket.listen(3)
        print("Waiting for connection from Algo...")

        self.client_sock, self.address = self.socket.accept()
        print("Connected to Algo @ " + str(self.address) + "!")

    # receive the first message from client, know the client address
    # print "Algo Connected"

    def disconnect(self):
        try:
            self.socket.close()
        except Exception as e:
            print("Algo disconnection exception: %s" % str(e))

    def write(self, msg):
        try:
            self.client_sock.sendto(msg.encode('utf-8'), self.address)
        except socket.error as e:
            if isinstance(e.args, tuple):
                print("errno is %d" % e[0])
                if e[0] == errno.EPIPE:
                    # remote peer disconnected
                    print("Detected remote disconnect")
                else:
                    # for another error
                    pass
            else:
                print("socket error ", e)
            sys.exit()
        except IOError as e:
            print("Algo read exception", e)
            print(traceback.format_exc())
            pass

    def read(self):
        try:
            msg = self.client_sock.recv(1024).decode()
            return msg
        except socket.error as e:
            if isinstance(e.args, tuple):
                print("errno is %d" % e[0])
                if e[0] == errno.EPIPE:
                    # remote peer disconnected
                    print("Detected remote disconnect")
                else:
                    # for another error
                    pass
            else:
                print("socket error ", e)
            sys.exit()

        except IOError as e:
            print("Algo read exception: ", e)
            print(traceback.format_exc())
            pass


if __name__ == '__main__':
    algo = Algo()
    try:
        number = 8
        counter = 1
        while True:
            time.sleep(5)
            if counter == number:
                algo.write("S" + str(counter)) # signal to stop
                print(algo.read())
                break
            else:
                algo.write("R" + str(counter)) # signal to take picture
                print(algo.read())
            counter += 1
    except KeyboardInterrupt:
        print("Terminating the program now...")
