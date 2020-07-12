import os
import sys
import time
import cv2
from openvino.inference_engine import IECore, IENetwork
import logging as log
import numpy as np
import argparse
import pyautogui
from face_detection import FaceDetection
from facial_landmarks_detection import LandmarksDetection
from head_pose_estimation import HeadPoseEstimation
from mouse_controller import MouseController
from base_model import BaseModel

#Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-m1", "--model1", required=True, type=str, help="Path to FaceDetection model")
ap.add_argument("-m2", "--model2", required=True, type=str, help="Path to LandmarksDetection model")
ap.add_argument("-m3", "--model3", required=True, type=str, help="Path to HeadPoseEstimation model")
ap.add_argument("-m4", "--model4", required=True, type=str, help="Path to GazeEstimation model")
ap.add_argument("-i", "--input", required=True, type=str, help="Path to image or video file")
ap.add_argument("-d", "--device", type=str, default="CPU", help="Specify the target device to infer on")
ap.add_argument("-o", "--output", type=str, default="output", help="output file for storing stats")
ap.add_argument("-pt", "--threshold", type=float, default=0.5, help="Probability threshold for detections filtering")
ap.add_argument("-fdv", "--fdv", type=str, help="FaceDetection visualization")
ap.add_argument("-lmv", "--lmv", type=str, help="LandmarksDetection visualization")
ap.add_argument("-hpv", "--hpv", type=str, help="HeadPoseEstimation visualization")
args = vars(ap.parse_args())

pyautogui.FAILSAFE = False

mltime_s = time.time()
#importing models
net1 = BaseModel(args['model1'])
m1time_s = time.time()
fd_shape, fd_name = net1.load_model()
face_load_time= time.time() - m1time_s
print('FaceDetection Model Load Time: {}'.format(face_load_time))

net2 = BaseModel(args['model2'])
m2time_s = time.time()
lm_shape, lm_name = net2.load_model()
land_load_time= time.time() - m2time_s
print('LandmarksDetection Model Load Time: {}'.format(land_load_time))

net3 = BaseModel(args['model3'])
m3time_s = time.time()
hp_shape, hp_name = net3.load_model()
head_load_time= time.time() - m3time_s
print('HeadPoseEstimation Model Load Time: {}'.format(head_load_time))

net4 = BaseModel(args['model4'])
m4time_s = time.time()
ge_shape, hp_name = net4.load_model()
gaze_load_time= time.time() - m4time_s
print('GazeEstimation Model Load Time: {}'.format(gaze_load_time))

print('Total Model Load Time: {}'.format((face_load_time+land_load_time+head_load_time+gaze_load_time)))


#mouse controller
mouse = MouseController('high', 'fast')

#handling input
if args['input'] == 'CAM':
	inputstream = 0 
	print('CAM')
elif args['input'].endswith('mp4') or args['input'].endswith('flv') or args['input'].endswith('avi'):
	inputstream = args['input']
	print('Video')
else:
	print('Input not supported')

cap = cv2.VideoCapture(inputstream)
cap.open(inputstream)
face_time = []
head_time = []
land_time = []
gaze_time = []


while cap.isOpened():

	### TODO: Read from the video capture ###
	flag, frame = cap.read()
	if not flag:
		break

	#FaceDetection
	face_inf, output_fd = net1.predict(frame.copy(), fd_shape)
	fd_image, fd_vis = FaceDetection.preprocess_output(output_fd[fd_name], frame.copy(), args['threshold'])
	face_time.append(face_inf)
	if args['fdv']:
		cv2.imshow('FD Vis', cv2.resize(fd_vis, (700, 500)))
		cv2.waitKey(1)
	if len(fd_image) == []:
		print('Face not detected')
		continue

	#FacialLandmarksDetection
	land_inf, output_lm = net2.predict(fd_image, lm_shape)
	left_eye, right_eye, ml_vis = LandmarksDetection.preprocess_output(output_lm[lm_name], fd_image)
	land_time.append(land_inf)
	if args['lmv']:
		cv2.imshow('FD Vis', ml_vis)
		cv2.waitKey(1)
	if len(left_eye) == []:
		print('Left eye not detected')
		continue
	elif len(right_eye) == []:
		print('Right eye not detected')
		continue
	
	#HeadPoseEstimation
	head_inf, output_hp = net3.predict(fd_image, hp_shape)
	p, r, y, hp_vis = HeadPoseEstimation.preprocess_output(output_hp, frame.copy())
	head_time.append(head_inf)
	if args['hpv']:
		cv2.imshow('FD Vis', cv2.resize(hp_vis, (700, 500)))
		cv2.waitKey(1)
	head_pose_angles = np.array([[y, p, r]])

	#GazeEstimation
	out, gaze_inf = net4.ge_predict(head_pose_angles, left_eye, right_eye, hp_shape)
	gaze_time.append(gaze_inf)
	mouse.move(out[0][0], out[0][1])

	
avg_face_time =  sum(face_time) / len(face_time)
avg_land_time =  sum(land_time) / len(land_time)
avg_head_time =  sum(head_time) / len(head_time)
avg_gaze_time =  sum(gaze_time) / len(gaze_time)

# Write load time, inference time, and fps to txt file
with open(f"output/{args['output']}.txt", "w") as f:
	f.write(str(avg_face_time)+'\n')     #Average Face Detection Inference Time
	f.write(str(1/avg_face_time)+'\n')   #Face Detection FPS
	f.write(str(sum(face_time))+'\n')    #Total Face Detection Inference Time

	f.write(str(avg_land_time)+'\n')     #Average Landmark detection Inference Time
	f.write(str(1/avg_land_time)+'\n')   #Landmark detection FPS
	f.write(str(sum(land_time))+'\n')    #Total Landmark detection Inference Time
    
	f.write(str(avg_head_time)+'\n')     #Average Head Pose Inference Time
	f.write(str(1/avg_head_time)+'\n')   #Head Pose FPS
	f.write(str(sum(head_time))+'\n')    #Total Head Pose Inference Time
    
	f.write(str(avg_gaze_time)+'\n')     #Average Gaze Detection Inference Time
	f.write(str(1/avg_gaze_time)+'\n')   #Gaze Detection FPS
	f.write(str(sum(gaze_time))+'\n')    #Total Gaze Detection Inference Time
cap.release()
cv2.destroyAllWindows()