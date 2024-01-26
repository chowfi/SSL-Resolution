
import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch.nn.functional as F
from tqdm import tqdm


class Autoencoder(nn.Module):
    def __init__(self, latent_channels=16, input_shape=(28,28)):
        """
        Autoencoder class for image data compression and reconstruction.

        Args:
            channels (int, optional): Number of input channels. Default is 4.
            depth (int, optional): Depth of the encoder and decoder, controlling
                the number of convolutional layers. Default is 3.
            conv_depth (int, optional): Depth of each convolutional layer (number of filters)

        Attributes:
            encoder (Encoder): The encoder part of the autoencoder.
            decoder (Decoder): The decoder part of the autoencoder.
        """
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_channels,input_shape)
        self.decoder = Decoder(latent_channels,input_shape)

    def forward(self, x):


        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Encoder(nn.Module):
    def __init__(self, latent_channels=16, input_shape=(28,28)):
        """
        Encoder class for the autoencoder, responsible for feature extraction.

        Args:
            channels (int): Number of input channels.
            depth (int, optional): Depth of the encoder, controlling the number
                of convolutional layers. Default is 3.
            conv_depth (int, optional): Depth of each convolutional layer (number of filters)
        """
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.AdaptiveAvgPool2d((14, 14)),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.AdaptiveAvgPool2d((7, 7)),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
      

class Decoder(nn.Module):
    def __init__(self,latent_channels=16,input_shape=(28,28)):
        """
        Decoder class for the autoencoder, responsible for image reconstruction.

        Args:
            channels (int): Number of output channels.
            depth (int, optional): Depth of the decoder, controlling the number
                of convolutional layers. Default is 3.
            conv_depth (int, optional): Depth of each convolutional layer (number of filters)
        """
        super(Decoder, self).__init__()
        
        if input_shape==[14,14]:
            padding=0
        else:
            padding=1

        decoder_layers = []
        decoder_layers.append(nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.ConvTranspose2d(64, 4, kernel_size=4, stride=2, padding=padding))

        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded
  
    
class FlatAutoencoder(nn.Module):
    def __init__(self, image_size, embedding_dim=128, channels=4):
        """
        Autoencoder class for image data compression and reconstruction.

        Args:
            channels (int, optional): Number of input channels. Default is 4.
            depth (int, optional): Depth of the encoder and decoder, controlling
                the number of convolutional layers. Default is 3.
            conv_depth (int, optional): Depth of each convolutional layer (number of filters)

        Attributes:
            encoder (Encoder): The encoder part of the autoencoder.
            decoder (Decoder): The decoder part of the autoencoder.
        """
        super(FlatAutoencoder, self).__init__()
        self.encoder = FlatEncoder(image_size,channels,embedding_dim)
        self.shape_before_flattening = self.encoder.shape_before_flatten(channels,image_size)
        self.decoder = FlatDecoder(embedding_dim,self.shape_before_flattening,channels,image_size)

    def forward(self, x):
        encoded = self.encoder(x)
        #print(f"encoded shape: {encoded.shape}")
        decoded = self.decoder(encoded)
        #print(f'decoded shape: {decoded.shape}')

        #return decoded
        return decoded


class FlatEncoder(nn.Module):
    def __init__(self, image_size, channels, latent_dim):
        super(FlatEncoder, self).__init__()

        # define convolutional blocks
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, latent_dim, kernel_size=3, stride=2, padding=1)
        
        # store shape before flatten
        self.shape_before_flattening = None

        # compute the flattened size after convolutions
        self.flattened_size = self.shape_after_flatten(channels,image_size)

        # define fully connected layer to create embeddings
        #self.fc = nn.Linear(self.flattened_size, latent_dim)
        self.fc = nn.Linear(latent_dim, latent_dim)
    
    def shape_before_flatten(self,channels,image_size):
        '''
        Calculate the shape of data before flattening to dense layers
        Used in decoder to structure reshape layer (1D -> 3D)
        '''
        x = torch.randn((1,channels,image_size,image_size))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # store the shape before flattening
        self.shape_before_flattening = x.shape[1:]
        return self.shape_before_flattening

    def shape_after_flatten(self,channels,image_size):
        '''
        Calculate shape of data after flattening from 3D to 1D
        Used to determine dense unit input shape (Conv -> Dense)
        '''
        
        x = torch.randn((1,channels,image_size,image_size))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # flatten the tensor
        x = x.view(x.size(0), -1)
        return list(x.shape[1:])[0]

    def forward(self, x):
        # move through conv blocks
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # flatten the tensor
        #x = x.view(x.size(0), -1)
    
        # apply global average pooling to the tensor
        x = torch.mean(x,dim=(2,3))

        # apply fully connected layer to generate embeddings
        x = self.fc(x)
        return x


class FlatDecoder(nn.Module):
    def __init__(self, latent_dim, shape_before_flattening, channels, image_size):
        super(FlatDecoder, self).__init__()

        # define fully connected layer to unflatten the embeddings
        self.fc = nn.Linear(latent_dim, np.prod(shape_before_flattening))
        
        # store the shape before flattening
        self.reshape_dim = shape_before_flattening
        
        # correct reconstruction shape
        if image_size == 14: padding=1
        else: padding=0
        # handle 2x2 image properly
        if image_size == 2:
            zero_pad = 2
        else:
            zero_pad = 1

        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=padding)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=zero_pad, output_padding=0)
        
        # define final convolutional layer to generate output image
        self.conv1 = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):

        # apply fully connected layer to unflatten the embeddings
        x = self.fc(x)
        
        # reshape the tensor to match shape before flattening
        x = x.view(x.size(0), *self.reshape_dim)
        # transpose conv blocks
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        # normal conv block at end with sigmoid activation
        x = torch.sigmoid(self.conv1(x))
        return x


def plot_autoencoder_reconstructions(model, val_loader, epoch, device, run_id=None):
    '''
    Plot train image next to reconstruction
    
    Args:
        model (Autoencoder): model to evaluate
        val_loader (DualLoader)

    '''

    model.eval()

    next_batch, _ = next(iter(val_loader))
    next_image = next_batch[0][0]
    fig, axs = plt.subplots(1,2,figsize=(8,4))

    #fig.suptitle(f"Autoencoder Results | Latent Dim = 256")


    # plotting full size data
    original_image = next_image[:3].permute(1,2,0)
    axs[0].imshow(original_image)
    axs[0].set_title(f"Original ({original_image.shape[0]}x{original_image.shape[1]})")


    prediction = model(next_batch[0].to(device))[0][:3].permute(1,2,0).to('cpu').detach()

    axs[1].imshow(prediction)
    axs[1].set_title(f"Reconstructed ({prediction.shape[0]}x{prediction.shape[1]})")

    # turn off tickmarks
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    if run_id is not None:
        plt.savefig(f"experiments/models/{run_id}/reconstruction_epoch{epoch}.png")
    else:
        plt.savefig(f"encoder_test_results/reconstuction_epoch_{epoch}.png")
    return


def train_autoencoder(autoencoder, train_loader, val_loader, num_epochs=10, loss="binary_crossentropy", run_id=None):
    '''
    Train an autoencoder on a data loader

    Args:
        autoencoder (Autoencoder): model to be trained
        train_loader (DataLoader): dataset to train on
        val_loader (DataLoader): dataset to validate on
        num_epochs: number of epochs to train
    '''


    # create directory for results if it doesn't exist
    if "encoder_test_results" not in os.listdir():
        os.makedirs("encoder_test_results")

    if loss=="binary_crossentropy":
        criterion = nn.BCELoss()
    elif loss=="mse":
        criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    epoch_losses = []
    val_epoch_losses = []

    print("Training autoencoder...")
    overall_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # train through epoch
        autoencoder = autoencoder.to(device)
        autoencoder.train()  # set model to training
        for images,y in tqdm(train_loader):
            inputs, _, _ = images
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs)  # bug: input 2x2 outputs 4x4
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = round(running_loss / len(train_loader), 5)
        epoch_losses.append(epoch_loss)

        # validate every epoch
        autoencoder.eval()  # set model to eval
        val_loss = 0.0
        with torch.no_grad():
          
            for images, y in val_loader:
                inputs, _, _ = images
                inputs = inputs.to(device)
                outputs = autoencoder(inputs)
                val_loss += criterion(outputs, inputs).item()
            epoch_val_loss = round(val_loss / len(val_loader), 5)     
        val_loss = round(val_loss / len(val_loader), 5)
        val_epoch_losses.append(val_loss)

        # log results and plot predictions by epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss}, Val Loss: {epoch_val_loss} Epoch Time: {round(time.time()-epoch_start_time,2)} seconds")
        plot_autoencoder_reconstructions(autoencoder,val_loader,epoch,device,run_id=run_id)

        # save model after every epoch
        if run_id is not None:
            torch.save(autoencoder,f"experiments/models/{run_id}/autoencoder_epoch_{epoch}.pth")
        else:
            torch.save(autoencoder,f"encoder_test_results/autoencoder_epoch_{epoch}.pth")

    data = {'epoch': range(1, len(epoch_losses) + 1),
            'train_loss': epoch_losses,
            'val_loss': val_epoch_losses}
    df = pd.DataFrame(data)
    if run_id is not None:
        df.to_csv(f"experiments/models/{run_id}/autoencoder_loss.csv")
    else:
        df.to_csv("encoder_test_results/loss.csv", index=False)

    print(f"Finished Training - Total Train Time = {round(time.time()-overall_start_time,2)}")
    return
