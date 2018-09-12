# [Medical Segmentation Decathlon](http://medicaldecathlon.com)

This is my source code for the medical decathlon, a generalizable 3D segmentation challenge. The objective of the competition is to develop a single segmentation model that can segment images of 10 different organs, namely, liver, brain, hippocampus, lung, prostrate, cardiac, pancreas, colon, hepatic vessels and spleen. 

#### Step 1: Data Exploration
##### [Data stats](https://github.com/suryatejadev/medseg_decathlon/blob/master/notebooks/data_exploration.ipynb)

#### Approach
My approach to this problem involved two steps. First, I trained a model to classify the organ of the input image. Using this output, I generated a conditional map, which is an image with the same intenstiy for all pixels. Each organ is assigned a specific intensity value. This map acts as a conditioning layer to the following segmentation model, and is appended to the input image. A multi-label segmentation model is then trained using the concatenated input image.  

#### Implementation
**Models Implemented:**  Dilated Densenet, UNet
**Training the model:** Train the model as follows:
```
Classification:
cd scripts
python classify.py --config=configs/config_classify.yaml

Segmentation:
cd scripts
python segment.py --config=configs/config_segment_densenet.yaml
```
 
