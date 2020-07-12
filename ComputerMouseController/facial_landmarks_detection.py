from base_model import BaseModel
import cv2

class LandmarksDetection(BaseModel):

    def __init__(self):

        Model.__init__(self) # initialize the base class
    
    def preprocess_output(outputs, image):

        h, w, c = image.shape
        x1min = int((outputs[0][0] * w) - 25)
        x1max = int((outputs[0][0] * w) + 25)
        y1min = int((outputs[0][1] * h) - 25)
        y1max = int((outputs[0][1] * h) + 25)
        x2min = int((outputs[0][2] * w) - 25)
        x2max = int((outputs[0][2] * w) + 25)
        y2min = int((outputs[0][3] * h) - 25)
        y2max = int((outputs[0][3] * h) + 25)
        if x1min < 0:
            x1min = 0
        left_eye = image[y1min:y1max, x1min:x1max]
        right_eye = image[y2min:y2max, x2min:x2max]
        cv2.rectangle(image, (x1min, y1min), (x1max, y1max), (255,0,0), 2)
        cv2.rectangle(image, (x2min, y2min), (x2max, y2max), (255,0,0), 2)

        return left_eye, right_eye, image