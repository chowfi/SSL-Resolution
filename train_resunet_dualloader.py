import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from simsiam import SimSiamModel # Replace with your actual SimSiam model implementation
from base_encoder import SimpleUNetEncoder
import torch.nn.functional as F
from dualloader2 import DualLoader
from base_encoder3 import Encoder
import csv
import torch.nn.init as init
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    np.random.seed(seed)


     


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



# Assuming input images are 3xHxW (3 channels, variable height and width)
#base_encoder = SimpleUNetEncoder(in_channels=4, out_channels=128)
# Set a specific seed value
seed_value = 42
set_seed(seed_value)
base_encoder=Encoder(latent_channels=128)



# Create SimSiam model
#simsiam_model = SimSiamModel(base_encoder,in_dim=128,projection_dim=2048)
simsiam_model=SimSiamModel(base_encoder,512,128)
#predictor=prediction_MLP()
initialize_weights(simsiam_model)
#initialize_weights(predictor)


# Print the SimSiam model architecture
print(simsiam_model)

batch_size=64
train_dataset = DualLoader(X_path="deepsat-sat6/X_train_sat6.csv", y_path="deepsat-sat6/y_train_sat6.csv",target_size=(2,2))
val_dataset = DualLoader(X_path="deepsat-sat6/X_test_sat6.csv", y_path="deepsat-sat6/y_test_sat6.csv",target_size=(2,2))

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

validation_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Print the first batch to check the shape
for batch in train_loader:
    images, labels = batch
    print("View 1 shape:", images[1].shape)
    print("View 2 shape:", images[2].shape)
    break

# Set up SimSiam training parameters
optimizer = optim.SGD(simsiam_model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-6)
num_epochs=35
scheduler = MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1) 


# Initialize a CSV file for logging training and validation loss
csv_filename = 'simsiam_training_log_lr_scheduler_14x14.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])




# Specify the checkpoint file to be loaded
restore=False
if restore== True:
    checkpoint_path = 'checkpoint_folder_lr_scheduler_upto_90th/simsiam_model_optimizer_epoch_10.pth'

   

    # Load the checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Check if the checkpoint file contains the necessary information
    if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
        # Load the model and optimizer states
        simsiam_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load additional training-related information if needed
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        val_loss = checkpoint['val_loss']

        print(f"Resuming training from epoch {start_epoch}, with loss: {loss}, and validation loss: {val_loss}")
    else:
        print("Checkpoint file does not contain the necessary information.")
else:
    start_epoch=0

simsiam_model.cuda()
# predictor.cuda()

# Training loop
for epoch in range(start_epoch,num_epochs):
    print('The epoch no progressing is...'+str(epoch))
    simsiam_model.train()  # Set the model to training mode

    # Training phase
    for data in train_loader:
        (images, y) = data
        _, view1, view2 = images  # Modify this based on your dataset structure
        view1, view2 = view1.cuda(), view2.cuda()

        # Forward pass
        z1, z2,p1, p2  = simsiam_model(view1, view2)
        #p1,p2 = predictor(z1), predictor(z2)

        # Compute loss # Negation because it's a minimization problem
        loss = D(p1, z2) / 2 + D(p2, z1) / 2

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation phase
    simsiam_model.eval()  # Set the model to evaluation mode
    total_validation_loss = 0.0
    with torch.no_grad():
        for val_data in validation_loader:
            val_images, val_y = val_data
            _, val_view1, val_view2 = val_images  # Modify this based on your dataset structure
            val_view1, val_view2 = val_view1.cuda(), val_view2.cuda()

            val_z1, val_z2,val_p1,val_p2 = simsiam_model(val_view1, val_view2)
            #val_p1, val_p2 = predictor(val_z1), predictor(val_z2)

            # Compute validation loss
            val_loss = D(val_p1, val_z2) / 2 + D(val_p2, val_z1) / 2
            total_validation_loss += val_loss.item()

    # Average validation loss
    average_validation_loss = total_validation_loss / len(validation_loader)

    # Print or log training and validation statistics
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item()}, Validation Loss: {average_validation_loss}')

    # Append values to the CSV file
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([epoch + 1, loss.item(), average_validation_loss])

    # Save the model at each epoch
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': simsiam_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'val_loss': average_validation_loss
    }, f'checkpoint_folder_lr_scheduler_14x14/simsiam_model_epoch_{epoch + 1}.pth')

