import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from model import MnistDataset, NeuralNetwork
from sklearn.metrics import f1_score, confusion_matrix

y_true = [0, 1, 0, 1]
y_pred = [1, 1, 1, 0]

# compute confusion matrix
cf = confusion_matrix(y_true, y_pred)
print(cf)

# flattening
cf = cf.reshape(-1)
tn, fp, fn, tp = cf
print(f"TP: {tp}. TN: {tn}. FP: {fp}. FN: {fn}")


print(os.listdir('C:/Users/andre/Desktop/Facultatea alternativa de inteligenta/Modulul 5/Dataset/MNIST_DATASET'))

training_ds_path = 'C:/Users/andre/Desktop/Facultatea alternativa de inteligenta/Materiale GitHub/CustomDataset_CNN/Dataset/MNIST_DATASET/training'
validation_ds_path = 'C:/Users/andre/Desktop/Facultatea alternativa de inteligenta/Materiale GitHub/CustomDataset_CNN/Dataset/MNIST_DATASET/testing'

# Generam numarul total de sample-uri din fiecare clasa si le salvam in distrib
distrib = [0] * 10
for idx, folder in enumerate(os.listdir(training_ds_path)):
    nr_of_imgs = len(os.listdir(os.path.join(training_ds_path, folder)))
    distrib[idx] = nr_of_imgs
# Facem plot pentru a vedea distributia claselor
print(distrib)
plt.title("Class Distribution")
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.bar(list(range(10)), distrib)
plt.show()


print(os.listdir(training_ds_path))
print(os.listdir(validation_ds_path))

# For reproductibility
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Hyperparameters
epochs = 15
batch_size_train = 32
learning_rate = 0.003

# Device used for training/inference
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device used: {device}')


# Dataloaderul nostru custom
train_dataset = MnistDataset('C:/Users/andre/Desktop/Facultatea alternativa de inteligenta/Materiale GitHub/CustomDataset_CNN/Dataset/MNIST_DATASET/training')
validation_dataset = MnistDataset('C:/Users/andre/Desktop/Facultatea alternativa de inteligenta/Materiale GitHub/CustomDataset_CNN/Dataset/MNIST_DATASET/testing')

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size_train, shuffle = True)
print(type(train_loader))
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 1, shuffle = False)

# Initiate the neural network
model = NeuralNetwork().to(device)

# Definirea loss-ului, functia NegativeLogLikeliHood
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Pipeline pentru antrenare si testare #

# Doua liste in care vom pastra erorile aferente fiecarei epoci
errors_train = []
errors_validation = []

# Training loop
for epoch in range(epochs):
    # O lista unde vom stoca erorile temporare epocii curente
    temporal_loss_train = [] 
    
    # Functia .train() trebuie apelata explicit inainte de antrenare
    model.train() 
    
    # Iteram prin toate sample-urile generate de dataloader
    for images, labels in train_loader:   
        images, labels = images.to(device), labels.to(device)

        # Clean the gradients
        optimizer.zero_grad()
        
        # Forward propagation
        output = model(images)

        # Compute the error
        loss = criterion(output, labels)
        temporal_loss_train.append(loss.item())
        
        # Backpropagation (computing the gradients for each weight)
        loss.backward()
        
        # Update the weights
        optimizer.step()
    
    # Now, after each epoch, we have to see how the model is performing on the validation set #
    # Before evaluation we have to explicitly call .eval() method
    model.eval()
    temporal_loss_valid = []
    for images, labels in validation_loader:
      images, labels = images.to(device), labels.to(device)

      # Forward pass
      output = model(images)
  
      # Compute the error
      loss = criterion(output, labels)
      temporal_loss_valid.append(loss.item())
      
    # Compute metrics after each epoch (mean value of loss) #
    medium_epoch_loss_train = sum(temporal_loss_train)/len(temporal_loss_train)
    medium_epoch_loss_valid = sum(temporal_loss_valid)/len(temporal_loss_valid)

    errors_train.append(medium_epoch_loss_train)
    errors_validation.append(medium_epoch_loss_valid)

    print(f"Epoch {epoch}. Training loss: {medium_epoch_loss_train}. Validation loss: {medium_epoch_loss_valid}")
    
    # Saving the model
    torch.save(model.state_dict(), f"{epoch}_model")


plt.title("Learning curves")
plt.plot(errors_train, label='Training loss')
plt.plot(errors_validation, label='Validation loss')
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.legend()
plt.show()
 
# Loading the model
loaded_model = NeuralNetwork().to(device)
loaded_model.load_state_dict(torch.load("14_model"))

# Computing evaluation metrics
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in validation_loader:
        images, labels = images.to(device), labels.to(device)

        test_output = loaded_model(images)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()

        y_true.append(labels.item())
        y_pred.append(pred_y.item())

scores = f1_score(y_true, y_pred, average=None)
print(f"f1 scores for each class: {scores}")

print(f"Mean f1: {round(np.mean(scores), 2)}%")

cf = confusion_matrix(y_true, y_pred)
        


# Afisarea numarului de parametri antrenabili ai modelului
model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Numarul total de parametri antrenabili ai modelului: {model_total_params}")