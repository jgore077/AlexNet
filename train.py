from AlexNet import AlexNet
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch
import sys
import os

EPOCH=int(sys.argv[1])
BATCH_SIZE=int(sys.argv[2])
LR=float(sys.argv[3])
MODEL_FOLDER=sys.argv[4]

os.makedirs(MODEL_FOLDER,exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],)
        ]
    )

    

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Declare model and move to the gpu
    cnn=AlexNet()
    cnn.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

    # Set model to train mode
    cnn.train()
    
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        val_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(tqdm(trainloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # Move inputs and labels to gpu
            inputs, labels =inputs.to(device),labels.to(device)
            
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            
            # Compute accuracy during training loop
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loss.backward()
            optimizer.step()
            
        # Compute train accuracy
        accuracy=100 * correct / total
        # Print training accuracy
        print(f'Epoch {epoch+1}: Train Accuracy: {accuracy :.2f}%')
        print(f"Testing model at epoch {epoch+1}")
        
        # Set model to evaluation mode
        cnn.eval()
        # Do testing without gradient descent
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs,labels in testloader:
                # Move inputs and labels to gpu
                inputs, labels =inputs.to(device),labels.to(device)
                outputs = cnn(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        # Compute accuracy
        accuracy=100 * correct / total
        
        # Print accuracy at epoch
        print(f'Epoch {epoch+1}: Test Accuracy: {accuracy :.2f}%')
        # Save the model out to a file 
        torch.save({
            'epoch':epoch,
            'model_state_dict':cnn.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
        },f"{MODEL_FOLDER}/e{epoch+1}_b{BATCH_SIZE}_a{int(accuracy)}.pt")
        
        # Reset the model to training mode    
        cnn.train()


    print('Finished Training')