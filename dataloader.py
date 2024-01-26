import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt

batch_size = 2

class DualLoader(Dataset):

    def __init__(self, X_path, y_path, target_size=(2, 2)):
        """
        DualLoader Dataset class for generating multiple transformations of image data.

        Attributes:
            data (ndarray): Raw image data loaded from CSV.
            y (Tensor): Corresponding labels for the image data.
            
        Args:
            X_path (str): Path to the CSV file containing image data.
            y_path (str): Path to the CSV file containing labels.
            target_size (tuple, optional): Desired output size for resized images. Default is (2, 2).
        """
        self.data = pd.read_csv(X_path).values
        self.y = torch.tensor(pd.read_csv(y_path).values).squeeze(1)
        
        # Transformation pipeline for the autoencoder
        self.autoencoder_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        
        #Transformation pipeline for autoencoder with original resolution.
        self.autoencoder_transform_originalres = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        # Transformation pipeline for the SimSiam (two different sets of augmentations: First set)
        self.simsiam_transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # First set of SimSiam transformation pipeline with original resolution.
        self.simsiam_transform1_originalres = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # Transformation pipeline for the SimSiam (two different sets of augmentations: Second set)
        self.simsiam_transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        # Second set of SimSiam transformation pipeline with original resolution.
        self.simsiam_transform2_originalres = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_array = self.data[idx].reshape(28, 28, 4).astype('uint8')
        
        ae_image = self.autoencoder_transform(image_array)
        sim_image1 = self.simsiam_transform1(image_array)
        sim_image2 = self.simsiam_transform2(image_array)

        ae_image_originalres = self.autoencoder_transform_originalres(image_array)
        sim_image1_originalres = self.simsiam_transform1_originalres(image_array)
        sim_image2_originalres = self.simsiam_transform2_originalres(image_array)
        
        
        # return (ae_image, ae_image_originalres, sim_image1, sim_image1_originalres, sim_image2, sim_image2_originalres), self.y[idx] # uncomment this line to see images before and after resolution changes below
        return (ae_image, sim_image1, sim_image2), self.y[idx]


# # Uncomment this entire section to see images before and after resolution changes

# data_iter = iter(train_loader)
# (images, y) = next(data_iter)

# (ae_images, ae_images_originalres,sim_images1, sim_images1_originalres, sim_images2, sim_images2_originalres) = images

# print(f"Autoencoder Images shape: {ae_images.shape}")
# print(f"SimSiam Images 1 shape: {sim_images1.shape}")
# print(f"SimSiam Images 2 shape: {sim_images2.shape}")
# print(f"Labels shape: {y.shape}")

# fig, axarr = plt.subplots(4, 3, figsize=(12, 8)) 

# axarr[0, 0].imshow(ae_images[0].permute(1, 2, 0).numpy())
# axarr[0, 0].set_title(f'Autoencoder Image 1\nLabel: {y[0].numpy()}')

# axarr[0, 1].imshow(sim_images1[0].permute(1, 2, 0).numpy())
# axarr[0, 1].set_title(f'SimSiam Image 1-1\nLabel: {y[0].numpy()}')

# axarr[0, 2].imshow(sim_images2[0].permute(1, 2, 0).numpy())
# axarr[0, 2].set_title(f'SimSiam Image 2-1\nLabel: {y[0].numpy()}')

# axarr[1, 0].imshow(ae_images_originalres[0].permute(1, 2, 0).numpy())
# axarr[1, 0].set_title(f'Autoencoder Image 1 (original res)\nLabel: {y[0].numpy()}')

# axarr[1, 1].imshow(sim_images1_originalres[0].permute(1, 2, 0).numpy())
# axarr[1, 1].set_title(f'SimSiam Image 1-1 (original res)\nLabel: {y[0].numpy()}')

# axarr[1, 2].imshow(sim_images2_originalres[0].permute(1, 2, 0).numpy())
# axarr[1, 2].set_title(f'SimSiam Image 2-1 (original res)\nLabel: {y[0].numpy()}')

# axarr[2, 0].imshow(ae_images[1].permute(1, 2, 0).numpy())
# axarr[2, 0].set_title(f'Autoencoder Image 2\nLabel: {y[1].numpy()}')

# axarr[2, 1].imshow(sim_images1[1].permute(1, 2, 0).numpy())
# axarr[2, 1].set_title(f'SimSiam Image 1-2\nLabel: {y[1].numpy()}')

# axarr[2, 2].imshow(sim_images2[1].permute(1, 2, 0).numpy())
# axarr[2, 2].set_title(f'SimSiam Image 2-2\nLabel: {y[1].numpy()}')

# axarr[3, 0].imshow(ae_images_originalres[1].permute(1, 2, 0).numpy())
# axarr[3, 0].set_title(f'Autoencoder Image 2 (original res)\nLabel: {y[1].numpy()}')

# axarr[3, 1].imshow(sim_images1_originalres[1].permute(1, 2, 0).numpy())
# axarr[3, 1].set_title(f'SimSiam Image 1-2 (original res)\nLabel: {y[1].numpy()}')

# axarr[3, 2].imshow(sim_images2_originalres[1].permute(1, 2, 0).numpy())
# axarr[3, 2].set_title(f'SimSiam Image 2-2 (original res)\nLabel: {y[1].numpy()}')

# plt.tight_layout()
# plt.show()

# [building, barren_land, trees, grassland, road, water]
