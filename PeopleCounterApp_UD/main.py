"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
from openvino.inference_engine import IECore, IENetwork
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    count = 0
    count_leave=0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
            count = 1
            if xmin>618 and ymin > 60 and xmax > 764:
                count_leave = 1
            
            
    return frame, count, count_leave


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    counter = 0
    person = 0
    detect = 0
    warning_timer = 0
    avg_time = 0
    image_mode = False
    # Initialise the class
    inference_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    n, c, h, w = inference_network.load_model(args.model, args.device, args.cpu_extension)
    
    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        inputstream = 0 
    elif args.input.endswith('jpg') or args.input.endswith('bmp') or args.input.endswith('png'):
        image_mode = True
        inputstream = args.input
    elif args.input.endswith('mp4') or args.input.endswith('flv') or args.input.endswith('avi'):
        inputstream = args.input
    else:
        print('Input not supported')
    
    
    cap = cv2.VideoCapture(inputstream)
    cap.open(inputstream)
    width = int(cap.get(3))
    height = int(cap.get(4))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        inf_start = time.time()

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (w, h))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        inference_network.async_inference(p_frame)

        ### TODO: Wait for the result ###
        if inference_network.wait() == 0:

            det_time = time.time() - inf_start
            ### TODO: Get the results of the inference request ###
            result = inference_network.extract_output()

            ### TODO: Extract any desired stats from the results ###
            frame, count, count_leave= draw_boxes(frame, result, args, width, height)
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
    
            #updating counter to display alarm if person stays longer then 20 sec    
            warning_timer +=1    
            
            #check if person is detected in frame
            if count == 1:
                detect +=1
            
            #if person is detected and he/she did not leave the frame
            if detect > 0 and count_leave == 0:
                count = 1
            
            #if person left the frame
            elif count_leave == 1:
                detect = 0
            
            #if no person in frame
            else:
                count = 0
            
            if detect == 1 and count_leave == 0:
                start_time = time.time()
            if count_leave == 1:
                person = person + count_leave
                client.publish("person", json.dumps({"total": person}))
                duration = int(time.time() - start_time)
                avg_time = duration
                client.publish("person/duration", json.dumps({"duration": duration}))
            
            #If person stays longer then 20 sec display message
            if avg_time > 20:
                message = "Person stayed longer then 20s"
                cv2.putText(frame, message, (15, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
           
            counter = count
            client.publish("person", json.dumps({"count": count}))

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        if warning_timer > 11:
            avg_time = 0
            warning_timer = 0
        

        ### TODO: Write an output image if `single_image_mode` ###
        if image_mode:
            cv2.imwrite('output_image.jpg', frame)
            
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    sys.stdout.flush()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    #args = build_argparser()
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
