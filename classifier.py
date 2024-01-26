import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from simsiam import *
from tqdm import tqdm
import time
import os


class Classifier(nn.Module):
    def __init__(self, input_dim=4096, num_classes=6, dropout_prob=0.2):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        #return F.softmax(x, dim=1)
        return x
    
class ConvClassifier(nn.Module):
    def __init__(self, image_height, image_width, in_channels, out_channels=4, num_classes=6, dropout_prob=0.5):
        super(ConvClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(image_height * image_width * out_channels, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)


def calculate_accuracy(predictions, labels):
    '''
    quick and dirty function to calculate accuracy given two one hot label vectors (predicted and true)
    '''
    _, predicted = torch.max(predictions, 1)
    _, true = torch.max(labels,1)
    correct = (predicted == true).type(torch.DoubleTensor).mean().item()
    return correct


def train_classifier(encoder, classifier, train_loader, val_loader, num_epochs=10, learning_rate=0.001, run_id=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    # create directory to store outputs
    if "classifier_test_results" not in os.listdir():
        os.makedirs("classifier_test_results")

    epoch_losses = []
    val_epoch_losses = []
    accuracies = []

    print("Training classifier...")
    overall_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # train
        
        encoder = encoder.to(device)
        classifier = classifier.to(device)
        classifier.train()
        for images, y in tqdm(train_loader):
            inputs, _,_= images
            inputs = inputs.to(device)
            optimizer.zero_grad()
            embeddings = encoder.encoder(inputs)
            y = y.float()
            y = y.to(device)
            outputs = classifier(embeddings)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = round(running_loss / len(train_loader), 5)
        epoch_losses.append(epoch_loss)

        # validate
        classifier.eval()
        val_loss = 0.0
        total_accuracy = 0.0
        with torch.no_grad():
            for images, y in val_loader:
                inputs, _,_= images
                inputs = inputs.to(device)
                embeddings = encoder.encoder(inputs)
                y = y.float()
                y = y.to(device)
                outputs = classifier(embeddings)
                loss = criterion(outputs, y)
                val_loss += criterion(outputs, y).item()
                total_accuracy += calculate_accuracy(outputs, y)
            epoch_val_loss = round(val_loss / len(val_loader), 5)
        val_loss = round(val_loss / len(val_loader), 5)
        accuracy = round(total_accuracy / len(val_loader),4)*100
        val_epoch_losses.append(val_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss}, Val Loss: {epoch_val_loss}, Val Acc: {accuracy}%, Epoch Time: {round(time.time()-epoch_start_time,2)} seconds")

        # save model after every epoch
        if run_id is not None:
            torch.save(classifier, f"experiments/models/{run_id}/classifier_epoch_{epoch}.pth")
        else:
            torch.save(classifier, f"classifier_test_results/classifier_epoch_{epoch}.pth")

    data = {'epoch': range(1, len(epoch_losses) + 1),
            'train_loss': epoch_losses,
            'val_loss': val_epoch_losses,
            'accuracy':accuracies}

    df = pd.DataFrame(data)
    if run_id is not None:
        df.to_csv(f"experiments/models/{run_id}/classifier_loss.csv")
    else:
        df.to_csv("classifier_test_results/loss.csv", index=False)

    print(f"Finished Training - Total Train Time = {round(time.time() - overall_start_time, 2)}")
    
    return

def evaluate_accuracy(encoder, classifier, test_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.eval()
    total_accuracy = 0.0
    
    with torch.no_grad():
        for images, y in test_loader:
            inputs, _,_= images
            inputs = inputs.to(device)
            embeddings = encoder.encoder(inputs)
            y = y.float().to(device)
            outputs = classifier(embeddings).to(device)
            total_accuracy += calculate_accuracy(outputs, y)
    accuracy = round(total_accuracy / len(test_loader),4)*100

    return accuracy
