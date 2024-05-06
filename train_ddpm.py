# Changed Diffusion trainer to integreate with VAE

import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from torch.utils.data import DataLoader
from unet_base import Unet
from linear_noise_scheduler import LinearNoiseScheduler

from VAE_with_decoder_input import VAE_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:4')

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    vae_config = config['vae_params']
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    
    dataset = ImageFolder(root=dataset_config['im_path'], transform=transforms.Compose([
        transforms.Resize((model_config['im_size'], model_config['im_size'])),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1)
    ]))
    data_loader = DataLoader(dataset=dataset, batch_size=train_config['batch_size'], shuffle=True)

    # Instantiate the model
    model = Unet(model_config).to(device)
    model.train()

    vae_config['input_size'] = len(dataset.class_to_idx)
    vae_config['output_size'] = model_config['im_channels'] * model_config['im_size'] * model_config['im_size']


    vae_model = VAE_Model(vae_config).to(device) 

    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'],train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ckpt_name']), map_location=device))
    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()
    
    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for im, class_id in tqdm(data_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)

            t750 = torch.full((im.shape[0],), 100).to(device)
            noisy_im750 = scheduler.add_noise(im, noise, t750).view(-1, vae_config['output_size'])
            class_id750 = torch.eye(vae_config['input_size'])[class_id].to(device)
            vae_model.fit(class_id750, noisy_im750)

            
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ckpt_name']))

        torch.save(vae_model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ckpt_name_vae']))

    
    print('Done Training ...')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='default.yaml', type=str)
    args = parser.parse_args()
    train(args)
