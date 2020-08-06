import os
import sys
import time
import json
import cv2
from openvino.inference_engine import IECore, IENetwork
import logging as log
from argparse import ArgumentParser
from inference import Network



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
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


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
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
                    
    return frame


def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    """
    image_mode = False
    video_mode = False


    # Initialise the class
    inference_network = Network()
    

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold


    #Loading the model
    n, c, h, w = inference_network.load_model(args.model, args.device, args.cpu_extension)
    

    #Handling the input stream
    if args.input == 'CAM':
        inputstream = 0 
    elif args.input.endswith('jpg') or args.input.endswith('bmp') or args.input.endswith('png'):
        image_mode = True
        inputstream = args.input
    elif args.input.endswith('mp4') or args.input.endswith('flv') or args.input.endswith('avi'):
        inputstream = args.input
    else:
        print('Input not supported')
    
    
    #initializing inputstream capture
    cap = cv2.VideoCapture(inputstream)
    cap.open(inputstream)
    width = int(cap.get(3))
    height = int(cap.get(4))

    
    #output video
    out = cv2.VideoWriter('output.mp4', 0x00000021, 24.0, (width,height))
    
    
    #Looping until stream is over
    while cap.isOpened():


        #Reading from the video capture
        flag, frame = cap.read()
       

        if not flag:
            break
        

        #key_pressed = cv2.waitKey(60)
        #inference start time
        inf_start = time.time()

        
        #Pre-processing the image as needed
        p_frame = cv2.resize(frame, (w, h))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        
        #Starting asynchronous inference for specified request
        inference_network.async_inference(p_frame)

        
        #Waiting for the result
        if inference_network.wait() == 0:

        
            #Inference end time
            det_time = time.time() - inf_start
            

            #Getting the results of the inference request
            result = inference_network.extract_output()

            
            #Extracting any desired stats from the results
            frame= draw_boxes(frame, result, args, width, height)
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            

        #writing video
        if video_mode:
        	print('writing')
        	fourcc = cv2.VideoWriter_fourcc(*'XVID')
        	out = cv2.VideoWriter('output.avi', fourcc, 24, (width,height))
        	out.write(frame)
    
            
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
