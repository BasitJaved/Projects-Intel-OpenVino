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

class BaseModel:
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
        self.input_name=None
        self.input_shape=None
        self.output_name= None
        self.output_shape=None


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.'''

        model=IENetwork(self.model_structure, self.model_weights)
        core = IECore()
        self.network = core.load_network(network=model, device_name=self.device, num_requests=1)
        self.check_model()
        self.input_name=next(iter(model.inputs))
        self.input_shape=model.inputs[self.input_name].shape
        self.output_name=next(iter(model.outputs))
        self.output_shape=model.outputs[self.output_name].shape
        return self.input_shape, self.output_name
     

    def predict(self, image, input_shape):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        prep_image = self.preprocess_input(image, input_shape)
        start_inference_time=time.time()#starting inference time
        self.network.start_async(request_id = 0,inputs = {self.input_name:prep_image})
        status = self.network.requests[0].wait(-1)
        if status == 0:
            outputs = self.network.requests[0].outputs
            faceinf = time.time()-start_inference_time
            
        return faceinf,outputs

    def check_model(self):
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")


    def preprocess_input(self, image, input_shape):
   
    #Before feeding the data into the model for inference,
    #you might have to preprocess it. This function is where you can do that.
    
        input_img=cv2.resize(image, (input_shape[3],input_shape[2])).transpose((2,0,1))
        input_img = input_img.reshape(1, *input_img.shape)
        
        return input_img
    
    def ge_predict(self, head_pose_angles, left_eye, right_eye, input_shape):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye = self.preprocess_input(left_eye, input_shape)
        right_eye = self.preprocess_input(right_eye, input_shape)
        net_input = {"head_pose_angles":head_pose_angles,"left_eye_image":left_eye,"right_eye_image":right_eye}
        start_inference_time=time.time()#starting inference time
        self.network.start_async(request_id = 0,inputs = net_input)
        status = self.network.requests[0].wait(-1)
        if status == 0:
            outputs = self.network.requests[0].outputs[self.output_name]
            gaze_inf = time.time()-start_inference_time
            
        return outputs, gaze_inf

        
