# bounding-box-regression

VGG16 architecture extrapolated the idea of greater depth in layers in the ILSVRC 2014. The model stood first for the object localization task and second for the object classification task. A slightly different objective pertaining to object detection
combines the classification task with the localization task, thus laying the foundation for identifying multiple relevant objects in a single image. Unlike classification models, these are more accurate and have specific applications. My VGG16 model has regression layers for predicting bounding boxes after feature extraction. The prediction network has been trained on PASCAL VOC dataset. I have been able to achieve close resemblance to the state-of-the-art results in terms of the IOU and mAP.

The VGG16 model can be summarized as below:
• The lower layers of VGG16 are trained on pre-trained weights to extract high-level features for fixed size input images.
• Extracted features are fed to the dense network with output as scaled bounding box coordinates. The ground truth boxes are scaled by the aspect ratio of each individual image.
• No region proposals or ROI-pooling layers involved in the training. The network learns to localize any arbitrary object from the ground truth box coordinates alone.


The input to the Convolutional Network is a fixed-size 224 X 224 X 3 image. The feature extraction step uses the pre-trained vgg16 model. The stack of convolutional filters is followed by 5 Fully Connected (FC) – layers. The first layer has 4096 neurons, second layer has 1024 neurons, third layer has 512 neurons, fourth layer has 100 neurons and fifth layer (output layer) has 4 neurons, each of which outputs one of the bounding box coordinates. All hidden layers are equipped with the rectification (Leaky Relu) non-linearity. Drop-out layers in between ensure that the model does not over-fit. Even though the weights are tuned to the classification model in standard VGG16 network, the high-level features are invariably advantageous for bounding box regression. 

Loss functions for a regression model are commonly Mean Squared Error, Mean Absolute Error, Log Cosh, Huber, and Quantile. MAE loss is useful when training data has lots of outliers but has uniform gradient for all error values. Huber loss is more robust to outliers
than MAE and MSE. Log-cosh is the logarithm of the hyperbolic cosine of the prediction error. It has the advantages of MSE and Huber loss and at the same time is twice differentiable everywhere, unlike Huber loss. As expected, for my experiments, Log-cosh
loss gave maximum accuracy. 

Choice of optimizer between Stochastic Gradient Descent (SGD) and RMSprop did not improve the accuracy by a significant amount given that the configurations of these optimizers ie. learning rate, momentum, and decay value remain the same. A low learning
rate of 0.001, momentum value of 0.9, and decay of 1e-6 gave the highest validation set accuracy. 

The bounding box cooridnates are read from the annotations files(.xml format) and stored in a text file using xmltocsv.py.
Training of the network is done using keras_vgg16.py.

The annotations file are read by xmltoxt.py and stored in dataset2.txt.
