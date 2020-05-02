# Monument-Recognition

It is an attempt to digitally preserve the monuments so that our future generation would also be aware of the glorious history of India. The main aim of this project is to analyze the archaeological monuments for its visual features to help in automating the process of identifying the monuments and to retrieve the similar images for studying art forms in greater details.

## Software and Tools Requirement

* Google Colaboratory- To accomplish all these above stated things the platform used is google’s Colab . It is a free cloud      service and now it supports free GPU!
* Machine Learning and Deep learning popular libraries Keras and scikit-learn.

## Process Used

1. Data Collection Phase- First of all images data is collected from different open source databases then preprocessing of data is done to convert data into desired format so that the model training and testing on the basis of image dataset becomes precise and accurate. Mostly, the famous Indian monuments were considered for dataset.

2. Feature Extraction Phase- Different feature Extractors(HOG Descriptor, LBP features etc.) are used initially for applying statistical model in the training phase but later the model replaced by Deep Learning Technique.

3. Model Training and Prediction- The selected data set is divided into training set and test set. Training set is used to train the machine and the test set is used to analyse the learning of the machine by predicting the classes of the given instances. The technique which will be applied on the data set is classification

4. Classifier which give highest accuracy(Convoluted Neural Network)- Neural Networks are essentially mathematical models to solve an optimization problem. They are made of neurons, the basic computation unit of neural networks. A neuron takes an input(say x), do some computation on it(say: multiply it with a variable w and adds another variable b ) to produce a value (say; z= wx+b). This value is passed to a non-linear function called activation function(f) to produce the final output(activation) of a neuron. There are many kinds of activation functions. One of the popular activation function is Sigmoid. Depending on the activation functions, neurons are named and there are many kinds of them like RELU, TanH etc. If you stack neurons in a single line, it’s called a layer; which is the next building block of neural networks.

## How CNN works?

![Neural Network Workflow](https://github.com/I-m-Parthian/Malware-Detection-Framework-for-a-virtual-environment/blob/master/Result/Neural%20Network.png)

Layers of Neural Network
* Convolution Layer- is a mathematical operation that’s used in single processing to filter signals, find patterns in signals etc. In a convolutional layer, all neurons apply convolution operation to the inputs, hence they are called convolutional neurons. The most important parameter in a convolutional neuron is the filter size.

* Pooling layer- is mostly used immediately after the convolutional layer to reduce the spatial size(only width and height, not depth). This reduces the number of parameters, hence computation is reduced. Also, less number of parameters avoid overfitting(don’t worry about it now, will describe it little later). The most common form of pooling is Max pooling where we take a filter of size F*F and apply the maximum operation over the F*F sized part of the image.

* Fully Connected Layer- If each neuron in a layer receives input from all the neurons in the previous layer, then this layer is called fully connected layer. The output of this layer is computed by matrix multiplication followed by bias offset.

