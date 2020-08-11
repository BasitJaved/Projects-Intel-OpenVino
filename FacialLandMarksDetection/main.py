import os
import sys
import time
import json
import cv2
from openvino.inference_engine import IECore, IENetwork
import logging as log
from argparse import ArgumentParser
from inference import Network
import numpy as np



def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m1", "--model1", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-m2", "--model2", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


# extracting width and height of face
def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 3)
            crop_img = frame[ymin:ymax, xmin:xmax]
            s_width = xmax - xmin
            s_height = ymax - ymin
                    
    return s_width, s_height, crop_img, xmin, ymin

#draw points for Facial Landmarks detection
def draw_points(frame, outputs, w, h, xmin, ymin):
    '''
    Draw bounding boxes onto the frame.
    
    
	'''

    for i in range(0, 70, 2):
    	x = int(outputs[0][i] * w) +xmin
    	y = int(outputs[0][i+1] * h) + ymin
    	cv2.circle(frame, (x, y), 1, (0, 255, 0), 10)
                    
    return frame


def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    """
    image_mode = False
    video_mode = False


    # Initialise the class
    inference_network1 = Network()
    inference_network2 = Network()
    

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold


    #Loading the model
    n1, c1, h1, w1 = inference_network1.load_model(args.model1, args.device)
    n2, c2, h2, w2 = inference_network2.load_model(args.model2, args.device)
    

    #Handling the input stream
    if args.input == 'CAM':
        inputstream = 0 
        video_mode = True
    elif args.input.endswith('jpg') or args.input.endswith('bmp') or args.input.endswith('png'):
        image_mode = True
        inputstream = args.input
    elif args.input.endswith('mp4') or args.input.endswith('flv') or args.input.endswith('avi'):
        inputstream = args.input
        video_mode = True
    else:
        print('Input not supported')
    
    
    #initializing inputstream capture
    cap = cv2.VideoCapture(inputstream)
    cap.open(inputstream)
    width = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter('output.mp4', 0x00000021, 24.0, (width,height))
    #Looping until stream is over
    while cap.isOpened():


        #Reading from the video capture
        flag, frame = cap.read()
       

        if not flag:
            break
        

        #inference start time
        inf_start = time.time()

        
        #Pre-processing the image as needed
        p_frame1 = cv2.resize(frame, (w1, h1))
        p_frame1 = p_frame1.transpose((2,0,1))
        p_frame1 = p_frame1.reshape(1, *p_frame1.shape)

        
        #Starting asynchronous inference for specified request
        inference_network1.async_inference(p_frame1)

        
        #Waiting for the result
        if inference_network1.wait() == 0:
            

            #Getting the results of the inference request
            result1 = inference_network1.extract_output()

            
            #Extracting face 
            s_width, s_height, crop_image, xmin, ymin = draw_boxes(frame, result1, args, width, height)


            #if no face detected in video
            if len(crop_image) == []:
            	print('Face Not Detected')
            	continue


            #Pre-processing the crop_image as needed
            p_frame2 = cv2.resize(crop_image, (w2, h2))
            p_frame2 = p_frame2.transpose((2,0,1))
            p_frame2 = p_frame2.reshape(1, *p_frame2.shape)


            #Starting asynchronous inference for second network (facial landmark detection)
            inference_network2.async_inference(p_frame2)


            #Waiting for the result
            if inference_network2.wait() == 0:

            	#Inference end time
            	det_time = time.time() - inf_start

            	#Getting the results of the inference request
            	result2 = inference_network2.extract_output()


            	#drawing points on face
            	frame= draw_points(frame, result2, s_width, s_height, xmin, ymin)


            #Extracting any desired stats from the results
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 255, 0), 2)


		
        
        #Writing an output image if single image was input
        if image_mode:
            cv2.imwrite('output_image.jpg', frame)
        else:
        	out.write(frame)
    

    cap.release()
    cv2.destroyAllWindows()
    sys.stdout.flush()


def main():
    """
    Load the network and parse the output.

    """
    # Grab command line args
    args = build_argparser().parse_args()
    

    # Perform inference on the input stream
    infer_on_stream(args)


if __name__ == '__main__':
    main()
