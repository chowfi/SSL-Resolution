from classifier_simsiam_op import Classifier,ConvClassifier,calculate_accuracy,train_classifier
from simsiam import SimSiamModel
from base_encoder3 import Encoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Subset
from torchvision import transforms, datasets
from dualloader2 import DualLoader



classifier=Classifier(input_dim=128)

base_encoder=Encoder(latent_channels=128)

# Create SimSiam model
#simsiam_model = SimSiamModel(base_encoder,in_dim=128,projection_dim=2048)
simsiam_model=SimSiamModel(base_encoder,512,128)
# checkpoint_path = 'checkpoint_folder_lr_scheduler_upto_90th/simsiam_model_epoch_68.pth'
# start_epoch = 0  # Initialize the starting epoch
# if torch.cuda.is_available():
#     checkpoint = torch.load(checkpoint_path)
# else:
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# print(checkpoint.keys())

#Check if the checkpoint file contains the necessary information
# if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
# simsiam_model.load_state_dict(checkpoint)
# simsiam_model.cuda()


checkpoint_path = 'checkpoint_folder_lr_scheduler_14x14/simsiam_model_epoch_30.pth'

   

# Load the checkpoint
if torch.cuda.is_available():
    checkpoint = torch.load(checkpoint_path)
else:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Check if the checkpoint file contains the necessary information
if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
    # Load the model and optimizer states
    simsiam_model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load additional training-related information if needed
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    val_loss = checkpoint['val_loss']

    print(f"Resuming training from epoch {start_epoch}, with loss: {loss}, and validation loss: {val_loss}")
else:
    print("Checkpoint file does not contain the necessary information.")
simsiam_model.cuda()

# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
train_dataset = DualLoader(X_path="deepsat-sat6/X_train_sat6.csv", y_path="deepsat-sat6/y_train_sat6.csv",target_size=(2,2))
val_dataset = DualLoader(X_path="deepsat-sat6/X_test_sat6.csv", y_path="deepsat-sat6/y_test_sat6.csv",target_size=(2,2))


# subsample dataset for data loader
subset_percentage = 1
num_samples_train = int(len(train_dataset) * subset_percentage)
num_samples_val = int(len(val_dataset) * subset_percentage)

train_loader = DataLoader(Subset(train_dataset, range(num_samples_train)), batch_size=32, shuffle=True,
                            num_workers=0)
val_loader = DataLoader(Subset(val_dataset, range(num_samples_val)), batch_size=32, shuffle=True,
                        num_workers=0)
#print(f"Data loaded in {round(time.time() - start_time, 2)} seconds")
train_classifier(simsiam_model,classifier,train_loader, val_loader, num_epochs=5, learning_rate=0.001)



