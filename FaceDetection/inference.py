import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        network = IENetwork(model=model_xml, weights=model_bin)
        
        # Get the supported layers of the network
        supported_layers = self.plugin.query_network(network=network, device_name="CPU")
        
        # Check for unsupported layers
        unsupported_layers = [l for l in network.layers.keys() if l not in supported_layers]
        
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
            
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network, device)
        
        # Get the input layer
        self.input_blob = next(iter(network.inputs))
        self.output_blob = next(iter(network.outputs))
        
        # Return the input shape (to determine preprocessing)
        return network.inputs[self.input_blob].shape

        
    def get_input_shape(self):
        
        return self.network.inputs[self.input_blob].shape

    
    def async_inference(self, image):
        
        
        self.exec_network.start_async(request_id = 0,inputs = {self.input_blob:image})
       
        return 
    

    def wait(self):
        
        status = self.exec_network.requests[0].wait(-1)
        return status


    def extract_output(self): #get_output
        
        return self.exec_network.requests[0].outputs[self.output_blob]
