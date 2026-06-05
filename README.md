# Cyient Project

## Done Work

1. The dataset chosen is a  [carotid artery dataset.](https://data.mendeley.com/datasets/d4xt63mgjm/1).
- This was done as I could not find a femoral artery dataset with ultrasound images. The assumption is that the model can be made to work on both these artery types with similar performance based on training data. The dataset contains 1100 ultrasound images and corresponding masks.
2. Finetuned a pretrained segmentation model  [SAM vit base](https://huggingface.co/facebook/sam-vit-base) from huggingface achieving IoU of 0.90 in the test set, and an F1 Score of 0.95. Here the encoder part is kept out of training
3. Converted the model to onnx format for efficiency (FP32 to INT8), with minimal loss in accuracy
4. Created a ROS package for the whole simulation flow. Currently a fallback image is used as feed to a pipeline that uses the vessel detector, and  manipulator is using the output from vessel detector to move locate its end-effector.

# TO DO

1. The work is pending to make the data from test set to be used as feed to the image_publher node, which gets published to image_raw topic
2. To use the actual manipulator than generated urdf
3. Make the model generated output in the vessel_detetor node.

# Difficulties/Assumptions
1. Since the output from the ultrasound is a cross-section of neck (should be leg, but in the dataset neck ultrasounds are used), there is no body-length positional data obtained here.(if x is along height of person, y is lateral to the person, and z-axis is from the eye upwards.) In true case, there should be a mapping between the ultrasound image pixel data to the US machine, and from the US machine to the manipulator
2. Because of 1., x and y have to be assumed here. Manipulator simply moves to center of the vessel in z-axis.
3. The ultrasound slice is also assumed to be strictly in the z-axis.

# DISCLAIMER
- Most of the work has been done using Claude Code