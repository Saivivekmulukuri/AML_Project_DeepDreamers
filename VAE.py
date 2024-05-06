# Completely self written for VAE with class label at encoder

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import cv2
import os
from tqdm import tqdm
import numpy as np
import torchvision
from torchvision.utils import make_grid
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def print_tensor(t):
    x_np = t.detach().cpu().numpy()
    np.set_printoptions(threshold=np.inf)
    print(x_np)

class VAE_Model(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, latent_size, decoder_hidden_size, output_size, learning_rate = 1e-3):
        super(VAE_Model, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(encoder_hidden_size, latent_size * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, decoder_hidden_size),
            nn.ReLU(),
            nn.Linear(decoder_hidden_size, decoder_hidden_size),
            nn.ReLU(),
            nn.Linear(decoder_hidden_size, output_size),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.class_to_id = {}
        self.current_class = 0

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        encodings = self.encoder(x)
        mu, log_var = encodings[:, :self.latent_size], encodings[:, self.latent_size:]
        z = self.reparameterize(mu, log_var)
        constructed_image = self.decoder(z)
        return constructed_image, mu, log_var
    
    def loss_function(self, constructed_image, x, mu, log_var):
        # recon_loss = nn.functional.binary_cross_entropy(constructed_image, x.view(-1, self.output_size), reduction='sum')
        recon_loss = nn.functional.mse_loss(constructed_image, x.view(-1, self.output_size), reduction='sum')
        # kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # return recon_loss + kld_loss
        return recon_loss
    
    def train(self, no_of_epochs, data_loader=False):
        for epoch in range(no_of_epochs):
            print("Epoch no: ", epoch)
            for images, class_id in tqdm(data_loader):

                images = images.to(device).view(-1, self.output_size)
                class_id = torch.eye(self.input_size)[class_id].to(device)

                constructed_image, mu, log_var = self(class_id)
                loss = self.loss_function(constructed_image, images, mu, log_var)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def sample(self,class_id,num_samples=1):
        self.eval()
        with torch.no_grad():
            class_id = torch.eye(self.input_size)[class_id].unsqueeze(0).to(device)
            samples = []
            for _ in range(num_samples):
                constructed_image, mu, log_var = self(class_id)
                samples.append(constructed_image)
            return torch.cat(samples, dim=0)
        

def MNIST_DataSet():

    num_channels = 3
    image_height = 28
    image_width = 28
    image_folder_path = './AML_Project/Main_Files/data/train/images'
    weights_file = './AML_Project/Main_Files/vae.ckpt'
    # image_folder_path = 'Main_Files/data/train/images'
    # weights_file = 'Main_Files/vae.ckpt'

    dataset = ImageFolder(root=image_folder_path, transform=transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1)
    ]))
    class_to_idx = dataset.class_to_idx
    batch_size = 128

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = VAE_Model(input_size=10, 
                      encoder_hidden_size=2048, 
                      latent_size=30, 
                      decoder_hidden_size=2048, 
                      output_size=num_channels*image_height*image_width, 
                      learning_rate=1e-3).to(device)

    
    model.train(no_of_epochs=1, data_loader=data_loader)
    torch.save(model.state_dict(), weights_file)


    state_dict = torch.load(weights_file, map_location=torch.device(device))
    model.load_state_dict(state_dict)

    for id,class_name in enumerate(class_to_idx):
        ims = model.sample(id,100)
        ims = ims.view(-1, num_channels, image_height, image_width)
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, 10)
        img = torchvision.transforms.ToPILImage()(grid)
        img.save('AML_Project/Main_Files/samples/class_id_{}.png'.format(id))
        img.close()

    


    # while True:
    #     class_id = input("Enter the class: ")
    #     class_id = class_to_idx[class_id]
    #     constructed_image = model.sample(class_id)
    #     out = constructed_image.view(-1, num_channels, image_height, image_width)
    #     image_np = out.squeeze().detach().cpu().numpy()
    #     image_np_uint8 = (image_np * 255).astype(np.uint8)
    #     cv2.imshow(f'Image of specified class', image_np_uint8)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


# def CatsAndDogs():

#     num_channels = 3
#     image_height = 128
#     image_width = 256
#     image_folder_path = "./AML_Project/Main_Files/Cats_Dogs/train"
#     weights_file = './AML_Project/Main_Files/cats_dogs.ckpt'
#     # image_folder_path = "Main_Files/Cats_Dogs/train"
#     # weights_file = 'Main_Files/cats_dogs.ckpt'
    
#     dataset = ImageFolder(root=image_folder_path, transform=transforms.Compose([
#         transforms.Resize((image_height, image_width)),
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: 2 * x - 1)
#     ]))
#     class_to_idx = dataset.class_to_idx
#     batch_size = 32

#     data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

#     model = VAE_Model(input_size=len(class_to_idx), 
#                       encoder_hidden_size=6400, 
#                       latent_size=1024,
#                       decoder_hidden_size=6400, 
#                       output_size=num_channels*image_height*image_width, 
#                       learning_rate=1e-3).to(device)

    
#     model.train(no_of_epochs=1000, data_loader=data_loader)
#     torch.save(model.state_dict(), weights_file)

#     # state_dict = torch.load(weights_file, map_location=torch.device(device))
#     # model.load_state_dict(state_dict)

#     # while True:
#     #     class_id = input("Enter the class: ")
#     #     if class_id not in class_to_idx:
#     #         continue
#     #     class_id = class_to_idx[class_id]
#     #     constructed_image = model.sample(class_id)
#     #     out = constructed_image.view(-1, num_channels, image_height, image_width)
#     #     image_np = out.detach().cpu().numpy()
#     #     image_np_uint8 = (image_np * 255).astype(np.uint8)
#     #     image_np_uint8 = image_np_uint8.squeeze(0)
#     #     image_np_uint8 = image_np_uint8.transpose(1, 2, 0)
#     #     cv2.imshow(f'Image of specified class', image_np_uint8)
#     #     cv2.waitKey(0)
#     #     cv2.destroyAllWindows()


if __name__ == '__main__':

    torch.manual_seed(42)
    MNIST_DataSet()
    # CatsAndDogs()