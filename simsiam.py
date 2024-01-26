import torch
import torch.nn as nn
import torch.nn.functional as F
# Example usage with a ResNet-based encoder
from torchvision.models import resnet50
from encoder
import torch.optim as optim
from torchvision import transforms, datasets
import csv
import torch.nn.init as init
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPHead, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
    
    
        x = F.relu(self.bn1(self.linear1(x)))
        
    
        x = self.linear2(x)
    
   
        return F.normalize(x, dim=-1)


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128, out_dim=512): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiamModel(nn.Module):
    def __init__(self, base_encoder, projection_dim=512,in_dim=16):
        super(SimSiamModel, self).__init__()
        self.encoder = base_encoder  # Your base encoder (ResNet, etc.)
        self.projector = MLPHead(in_dim, projection_dim)
        self.predictor=prediction_MLP()
        

    def forward(self, view1, view2):
        z1 = self.projector(self.encoder(view1))
        z2 = self.projector(self.encoder(view2))
        p1,p2=self.predictor(z1),self.predictor(z2)
        return z1, z2,p1,p2

# class SimSiam(nn.Module):
#     def __init__(self, backbone=resnet50()):
#         super().__init__()
        
#         self.backbone = backbone
#         self.projector = projection_MLP(backbone.output_dim)

#         self.encoder = nn.Sequential( # f encoder
#             self.backbone,
#             self.projector
#         )
#         self.predictor = prediction_MLP()
    
#     def forward(self, x1, x2):

#         f, h = self.encoder, self.predictor
#         z1, z2 = f(x1), f(x2)
#         p1, p2 = h(z1), h(z2)
#         L = D(p1, z2) / 2 + D(p2, z1) / 2
#         return {'loss': L}


# input_tensor = torch.randn(64, 4, 28, 28)

# encoder=Encoder(latent_channels=128)
# model=SimSiamModel(encoder,512,128)
# op=model(input_tensor,input_tensor)

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

def train_simsiam(autoencoder, train_loader,val_loader, num_epochs=10, run_id=None):
    seed_value = 42
    set_seed(seed_value)
    
    simsiam_model = SimSiamModel(autoencoder, 512, 128)
    initialize_weights(simsiam_model)
    optimizer = optim.SGD(simsiam_model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-6)
    # scheduler = MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1) 

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
            for val_data in val_loader:
                val_images, val_y = val_data
                _, val_view1, val_view2 = val_images  # Modify this based on your dataset structure
                val_view1, val_view2 = val_view1.cuda(), val_view2.cuda()

                val_z1, val_z2,val_p1,val_p2 = simsiam_model(val_view1, val_view2)
                #val_p1, val_p2 = predictor(val_z1), predictor(val_z2)

                # Compute validation loss
                val_loss = D(val_p1, val_z2) / 2 + D(val_p2, val_z1) / 2
                total_validation_loss += val_loss.item()

        # Average validation loss
        average_validation_loss = total_validation_loss / len(val_loader)

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
    
    return simsiam_model
    


