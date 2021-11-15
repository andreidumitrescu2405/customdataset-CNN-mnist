# customdataset-CNN-mnist
Custom data set implementation of recognizing hand-written digits using Convolutional Neural Networks

The Deep Learning algorithm used for recognizing hand-written digits using CNN. Our _model.py_ contains 2 classes:
  -MnistDataset which helps us get and load our dataset samples, our features and labels
  -NeuralNetwork which helps creating the Convolutional Neural Networks with 2 methods in it, ____init____ and __forward__. Our init method have the use of declaring the convolutional layers and also the last two final basic neural network layers, activations function such as LeakyReLU() and LogSoftMax() and batch normalization layers (nn.BatchNorm2d(), using pytorch). Our forward method helps moving from the Input layer (left) to the Output layer (right) in the neural network and that's called Forward Propagation.
Our _main.py_ file its used for training, validation and testing pipelines and also declaring our metrics, hyperparameters, initiating the neural network and plotting our results using metrics such as f1_score and confusion_matrix from sklearn.metrics

In order to run the code be sure that you're having both the files and the custom data set in the same directory and just call the command "python main.py" inside the command line interpreter.
