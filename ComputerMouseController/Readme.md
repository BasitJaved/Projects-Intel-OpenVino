##Computer Pointer Controller

CPC is an AI app which is used to control mouse pointer. In this app mouse pointer is controlled using head position and movement of eyes. Four intel edge AI models 
were used in creation of this app, and it was created using Intel OpenVino Toolkit. Input to this app can be a video file or live feed using webcam. The first model 
(Face detection model) detects the head in the video, this detection is then used as input to second (Landmarks regression model) and third model (Head Pose 
Estimation model), the second model detects the eyes on face and third model detects the orientation of head, output of second and third model is used as input to 
last model (Gaze estimation model) which estimates the coordinates of where person is looking, these coordinates are used as input to move mouse pointer using 
pyautogui library.
