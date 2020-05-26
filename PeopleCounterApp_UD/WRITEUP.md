# Project Write-Up



## Explaining Custom Layers

Any layer not in the list of supported framework layers of OpenVino toolkit is custom layer.
In Tensorflow to add the custom layer first option is to register the custom layer as extensions to the Model Optimizer.
Second option is to actually replace the unsupported subgraph with a different subgraph.
Final option is to actually offload the computation of subgraph back to tensorflow during inference

The model I chose was SSD Inception V2 COCO (ssd_inception_v2_coco_2018_01_28) from Tensorflow pretrained model zoo (https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html),

The command I used to convert the model to IR is 
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py 
--input_model frozen_inference_graph.pb 
--tensorflow_object_detection_api_pipeline_config pipeline.config  
--tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json 
--reverse_input_channels

Any unsupported subgraph in model was replaced with a supported subgraph using --tensorflow_use_custom_operations_config and passing ssd_v2_support.json file to it.

## Comparing Model Performance

In terms of performance before and after conversion, size of model is reduced significantly, total inference time on whole video is also reduced more than half but accuracy remains same in both cases.

#Before conversion
Size					294 MB (all files)
Inference time (complete video)		432.2 sec
Accuracy				70% (no detection/bounding box on a number of frames)

After conversion (IR)
Size					95.5 MB (bin, xml)
Inference time (complete video)		165.1 sec
Accuracy				70% (no visible change in accuracy)



## Assess Model Use Cases

1.	Retail: Retail is one of the best use cases of people counter app, it can be used to
a)	Check the stocks of different product and alert the management if a product is running low in stock on display
b)	It can be used to count the number of customers coming to store daily/weekly/monthly as well as during a specific time on any day or during certain event, if we combine it with facial detection we can use it to identify regular customers, times when they visit the store etc. and using this data management can decide how shopping experience can be improved for customers.
c)	It can be used to track the buying behavior of customers, which product are they buying most? Is it easy to find those product? In case customers are buying multiple products, which 2 or 3 products are they buying together most and what will be impact on sales if we put these products together.
d)	App along with facial recognition can be used to monitor attendance in office school, first entry in the morning can be logged as entry time while last entry on a specific date can be considered as leaving time for office and school.
2.	Safety/Security: Safety and security is another field where this app has wide use cases
a)	It can be used to monitor number of people coming in a certain building.
b)	Together with facial recognition, it can be used to restrict the entry of employees from a certain building or room in an organization, it can be used to track movement of employees in sensitive rooms/buildings and restrict their entry after a certain period of time or after accessing for a certain number of times.
c)	At home it can be used at the main gate to give access to family members only, it can be used to track children from going outside the main gate, it can be used to alert home owner and law enforcement in case there is an event of force entry when owner is not at home or during night when family is sleeping.
d)	At my country there are a lot of cases of car/motorcycle theft so this app along with facial recognition can be used to alert owner and law enforcement in case someone tries to steal a car, there will be no problem if owner of family tries to enter the car, but if owner is not nearby and someone else tries to get into car the app will generate alert to owner and if car starts to move away alert will be generated to owner as well as law enforcement.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size will affect the performance of model,
a)	Accuracy of model is not very good in optimal lighting conditions and decreased lightning will further decrease the accuracy.
b)	The model used in this app was taken from Tensorflow pre-trained model list and later converted to IR format, so accuracy is not very good, for better accuracy we can use model from OpenVino pre-trained model zoo (person-detection-retail-0002 or person-detection-retail-0013), these models gave almost 100% accuracy during testing phase.
c)	Focal Length/Image Size will effect performance if image is distorted or size of image is very small, if image size is equal to or greater then 300x300 and image is not distorted, then there will not be a major change in model performance.
