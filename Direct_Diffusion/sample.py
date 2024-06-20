import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from diffusion import unet
from diffusion import NoiseScheduler


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

def sample(model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
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
    model = unet(1).to(device)
    model.load_state_dict(torch.load(os.path.join("Direct_Diffusion/out",
                                                  "m1.pth"), map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = NoiseScheduler(steps=diffusion_config['num_timesteps'],
                                     noise_start=diffusion_config['beta_start'],
                                     noise_end=diffusion_config['beta_end'])
    scheduler.to(device)
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)