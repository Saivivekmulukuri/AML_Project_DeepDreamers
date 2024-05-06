# Changed Diffusion sampler to integreate with VAE

import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from unet_base import Unet
from linear_noise_scheduler import LinearNoiseScheduler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from VAE_with_decoder_input import VAE_Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:4')


def sample(model, scheduler, train_config, model_config, diffusion_config, vae_model, vae_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    for class_id in range(vae_config['input_size']):
        xt = vae_model.get_sample(class_id, train_config['num_samples'], device)
        xt = xt.view(-1, model_config['im_channels'], model_config['im_size'], model_config['im_size'])
        for i in tqdm(reversed(range(101))):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
            
            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
            
            # Save x0
            if i == 0:
                ims = torch.clamp(xt, -1., 1.).detach().cpu()
                ims = (ims + 1) / 2
                grid = make_grid(ims, nrow=train_config['num_grid_rows'])
                img = torchvision.transforms.ToPILImage()(grid)
                if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
                    os.mkdir(os.path.join(train_config['task_name'], 'samples'))
                img.save(os.path.join(train_config['task_name'], 'samples', f'classid_{4-class_id}x0_{i}.png'))
                img.close()

    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        if i == 0:
            ims = torch.clamp(xt, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            grid = make_grid(ims, nrow=train_config['num_grid_rows'])
            img = torchvision.transforms.ToPILImage()(grid)
            if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
                os.mkdir(os.path.join(train_config['task_name'], 'samples'))
            img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
            img.close()



def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    model.eval()

    dataset_config = config['dataset_params']
    dataset = ImageFolder(root=dataset_config['im_path'], transform=transforms.Compose([
        transforms.Resize((model_config['im_size'], model_config['im_size'])),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1)
    ]))
    vae_config = config['vae_params']
    vae_config['input_size'] = len(dataset.class_to_idx)
    vae_config['output_size'] = model_config['im_channels'] * model_config['im_size'] * model_config['im_size']

    vae_model = VAE_Model(vae_config).to(device) 

    state_dict = torch.load(os.path.join(train_config['task_name'],train_config['ckpt_name_vae']), map_location=torch.device(device))
    vae_model.load_state_dict(state_dict)

    
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config, vae_model, vae_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
