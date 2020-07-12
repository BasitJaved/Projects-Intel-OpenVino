from base_model import BaseModel
import cv2

class HeadPoseEstimation(BaseModel):

    def __init__(self):

        Model.__init__(self) # initialize the base class
    
    def preprocess_output(outputs, frame):
    
    #Before feeding the output of this model to the next model,
    #you might have to preprocess the output. This function is where you can do that.
    
        p = outputs["angle_p_fc"][0][0]
        r = outputs["angle_r_fc"][0][0]
        y = outputs["angle_y_fc"][0][0]
        cv2.putText(frame, "Pose Angles: pitch:{:.2f} , roll:{:.2f} , yaw:{:.2f}".format(p,r,y), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        return p, r, y, frame