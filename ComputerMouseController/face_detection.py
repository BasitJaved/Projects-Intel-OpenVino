from base_model import BaseModel
import cv2

class FaceDetection(BaseModel):

	def __init__(self):

		Model.__init__(self) # initialize the base class

	def preprocess_output(outputs, image, threshold):

		c_box = []

		for box in outputs[0][0]: # Output shape is 1x1x100x7
			conf = box[2]
			if conf > 0.5:
				xmin = int(box[3] * image.shape[1])
				ymin = int(box[4] * image.shape[0])
				xmax = int(box[5] * image.shape[1])
				ymax = int(box[6] * image.shape[0])
				crop_img = image[ymin:ymax, xmin:xmax]
				cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)

		return crop_img, image