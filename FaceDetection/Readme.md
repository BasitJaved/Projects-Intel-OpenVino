Face Detection using Intel OpenVino Toolkit. The code is written in Python and it uses Intel's OpenVINO toolkit to perform inference on input (Video/image).

## Directory Structure

  |---main.py
  
  |---Inference.py
  
  |---vid8.mp4
  
  |---output.mp4
  
  |---1.jpg
  
  |---output_image.jpg
  
  |---preview.jpg
  
  |---README.md

  |---models
  
    |---face-detection-retail-0005.bin
    |---face-detection-retail-0005.xml

## Demo

python main.py --model face-detection-retail-0005.xml --input 1.jpg
python main.py --model face-detection-retail-0005.xml --input vid1.mp4


## Output Video

Click the image below to view the output video

[![Output Video](preview.jpg)](https://www.youtube.com/embed/i9VRocFl-3w)
