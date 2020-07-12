'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import logging as log
import cv2
import time
from openvino.inference_engine import IECore, IENetwork

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model = model_name
        self.model_structure = model_name + '.xml'
        self.model_weights=model_name +'.bin'
        self.device = device
        self.extensions = extensions
        self.network = None
        self.input_names=None
        self.input_shape=None
        self.output_name= None
        self.output_shape=None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model=IENetwork(self.model_structure, self.model_weights)
        core = IECore()
        self.network = core.load_network(network=model, device_name=self.device, num_requests=1)  
        self.input_names=next(iter(model.inputs))
        self.input_shape=model.inputs[self.input_names].shape
        self.output_name=next(iter(model.outputs))
        self.output_shape=model.outputs[self.output_name].shape
        return

    def predict(self, head_pose_angles, left_eye, right_eye):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye, right_eye, head_pose_angles = self.preprocess_input(left_eye, right_eye, head_pose_angles)
        net_input = {"head_pose_angles":head_pose_angles,"left_eye_image":left_eye,"right_eye_image":right_eye}
        start_inference_time=time.time()#starting inference time
        self.network.start_async(request_id = 0,inputs = net_input)
        status = self.network.requests[0].wait(-1)
        if status == 0:
            outputs = self.network.requests[0].outputs[self.output_name]
            gaze_inf = time.time()-start_inference_time
            #coords = self.preprocess_outputs(outputs)

        return outputs, gaze_inf

    def check_model(self):
        
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")


    def preprocess_input(self, left_eye, right_eye, head_pose_angles):
    
    #Before feeding the data into the model for inference,
    #you might have to preprocess it. This function is where you can do that.
    
        
        left_eye=cv2.resize(left_eye, (60,60)).transpose((2,0,1))
        left_eye = left_eye.reshape(1, *left_eye.shape)
        

        right_eye=cv2.resize(right_eye, (60,60)).transpose((2,0,1))
        right_eye = right_eye.reshape(1, *right_eye.shape)
        
        return left_eye, right_eye, head_pose_angles

    #def preprocess_output(self, outputs):
    
    #Before feeding the output of this model to the next model,
    #you might have to preprocess the output. This function is where you can do that.
    #
     #   raise NotImplementedError
