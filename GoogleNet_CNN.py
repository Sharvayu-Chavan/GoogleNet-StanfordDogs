import torch
import torchvision.models as models
import os
from PIL import Image
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

test_losses = []
test_accuracies = []
epoch_numbers = []



googlenet = models.googlenet(pretrained=True)
# Replace the final classifier layer(s)
num_classes = 120  # Replace with the number of classes in your dataset
googlenet.fc = torch.nn.Linear(googlenet.fc.in_features, num_classes)

model = googlenet.to(device='cpu')
#main archive folder
folder_path = 'C:\\Programming Work\\PythonWork\\GoogleNet-StanfordDogs\\archive\\cropped'

trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

train_dataset = datasets.ImageFolder(root=folder_path + '\\train', transform=trans)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(root=folder_path + '\\test', transform=trans)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)


learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Define Training Loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        #Create prediction and loss from Forward propagation
        pred = model(X)
        loss = loss_fn(pred,y)
        
        #Backpropagation Portion
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader,model,loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_losses.append(test_loss)
    test_loss /= num_batches
    correct /= size
    test_accuracies.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



#Define Training Iterations
epochs = 30
for epoch in range(epochs):
    epoch_numbers.append(epoch)
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

plt.plot(epoch_numbers, test_losses, label='Average Test Losses')  # Plot first line
plt.plot(epoch_numbers, test_accuracies, label='Test Accuracies')  # Plot second line
plt.xlabel('Epoch Number')
plt.ylabel('Test Losses and Accuracies')
plt.title('Test Losses/Accuracies vs Epoch Count')
plt.legend()  # Add legend to distinguish between lines
plt.grid(True)
plt.show()

print("Done!")

#save model
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#loading model
model.load_state_dict(torch.load("model.pth"))
print("Loaded PyTorch Model State from model.pth")