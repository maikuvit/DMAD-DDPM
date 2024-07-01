import os
from torch.optim import Adam
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torch
import torch.nn as nn
from PIL import Image

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np

import datetime
import requests
import yaml 
import argparse

from pathlib import Path


#define all the parameters from the config file
BATCH_SIZE = None
EPOCHS = None
URL = None
OUTPUT_PATH = None
DATASET_PATH = None
PRETRAINED_PATH = None
device = None


def increasingLoss(losses):
    if len(losses) < 3:
        return False

    last_three = losses[-3:]
    return last_three[0] < last_three[1] < last_three[2]
    
def send_webHook(url, text):
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    response = requests.get(url, data=f"at time {current_time_str} -> {text}")

    if response.status_code == 200:
        print("OK")
    else:
        print("Error")

class autoencoder(nn.Module):
    
    def __init__(self, encoder):
        super().__init__()
        
        self.encoder = encoder
        
        self.latent_space = nn.Sequential( # this is to have a flatten representation of the latent space ...
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
             
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3,stride=2, padding=1,  output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent_space(x)
        x = torch.reshape(x, (x.shape[0], 256, 56, 56))
        x = self.decoder(x)
        return x
    
    def freeze_encoder(self,freeze = True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def get_embeddings(self, x):
        x = self.encoder(x)
        x = self.latent_space(x)
        return x
    
    def embeddings_to_out(self,x):
        x = torch.reshape(x, (x.shape[0], 256, 56, 56))
        x = self.decoder(x)
        return x
    
    def freeze_batchNorm(self, Freeze = True):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_ = not Freeze
    
def evaluate_epoch(model,dataset,device):
    model.eval()

    ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)
    mse = nn.MSELoss().to(device)
    running_vloss = 0.0

    with torch.no_grad():
        for img in (dataset):
            img = img.to(device)
            out = model(img).to(device)
            vloss = 1 - ssim(out, img) + mse(out, img)
            running_vloss += vloss
    return running_vloss / len(dataset)
    
# SOURCE: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
# let's start the training loop ...

def train(model, dataset, eval_set):
    
    dataset = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_set = DataLoader(eval_set, batch_size=BATCH_SIZE, shuffle=True)

    # Create output directories
    if not os.path.exists("out"):
        os.mkdir("out")
    
    learning_rate = 0.01
    last_epoch_change = 0
    # Specify training parameters
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # learning rate scheduler to decrease it gradually ...
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    early_stopper = EarlyStopper(patience=3, min_delta=0.001)

    ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)
    mse = nn.MSELoss().to(device)
    
    # Run training
    for epoch_idx in range(EPOCHS):
        
        model.train(True)
        
        losses = []
        eval_losses = []
        
        if device.type == 'cuda':
             torch.cuda.empty_cache()

        lastimg = None
        lastReco = None

        for image in tqdm(dataset):

            optimizer.zero_grad()
            image = image.float().to(device)

            reconstructed = model(image).to(device)


            loss = 1 - ssim(image, reconstructed) + mse(image, reconstructed) #combining mse and ssim loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            lastimg = image
            lastReco = reconstructed
        
        # check last three epochs loss, if the optimizer is not converging, decrease the learning rate ...
        if (epoch_idx - last_epoch_change > 2) and increasingLoss(losses) and learning_rate > 0.0001:
            learning_rate = learning_rate * 0.1
            optimizer = Adam(model.parameters(), lr=learning_rate)
            # learning rate scheduler to decrease it gradually ...
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
            last_epoch_change = epoch_idx

        else:
            scheduler.step()

        eval_loss = evaluate_epoch(model, eval_set, device)
        eval_losses.append(eval_loss)

        result = ('Finished epoch:{} | Loss : {:.4f} | Learning Rate: {} | Eval loss: {}'.format(
            epoch_idx + 1,
            np.mean(losses),
            learning_rate,
            eval_loss
        ))

        send_webHook(URL, result)
        print(result)

        torch.save(model.state_dict(), os.path.join(OUTPUT_PATH,
                                                    "ae_Casia_{}.pth".format(epoch_idx)))
        if early_stopper.early_stop(eval_loss):
            print('Early stopping')
            break
    
    print('Done Training ...')


class CasiaDataset(Dataset):
    r"""
    Dataset class to load the Bonafide images. 
    """
    def __init__(self, im_path):
        
        self.images = self.load_images(im_path)
        
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path: file with list of absolute paths to images 
        :return:
        """
        images = []
        assert os.path.isfile(im_path), "images path file {} does not exist".format(im_path)
        
        with open(im_path, 'r') as f:
            for line in f:
                images.append(line.strip())  
  
        print('Found {} images.'.format(len(images)))
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        im = Image.open(self.images[index])  
        im_tensor = torchvision.transforms.ToTensor()(im)
        # Convert input to -1 to 1 range.
        #im_tensor = (2 * im_tensor) - 1
        return im_tensor


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    args = parser.parse_args()
    conf = yaml.safe_load(Path(args.conf).read_text())
    
    BATCH_SIZE = conf['autoencoder']['batch_size']
    EPOCHS = conf['autoencoder']['epochs']
    URL = conf['autoencoder']['webhook']
    OUTPUT_PATH = conf['autoencoder']['output_path']
    DATASET_PATH = conf['autoencoder']['dataset_path']
    PRETRAINED_PATH = conf['autoencoder']['pretrained_path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.vgg16(pretrained=False)
    model.load_state_dict(torch.load(PRETRAINED_PATH))
    encoder = model.features[:11]

    del model

    print('Using device:', device)

    model = autoencoder(encoder).to(device)
    model.load_state_dict(torch.load('out_ae/ae_Casia_2.pth'))

    model.freeze_encoder()
    ds = CasiaDataset(DATASET_PATH)

    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

    starting_message = f"\nStarting training with {len(train_ds)} training samples and {len(test_ds)} test samples.\n~~ Parameters ~~\nBatch size: {BATCH_SIZE}\nEpochs: {EPOCHS}\nDevice: {device}\n"

    send_webHook(URL, starting_message)
    del starting_message
    try:
        train(model, train_ds, test_ds)
        send_webHook(URL, "Training completed")
    except Exception as e:
        print(e)
        send_webHook(URL, f"Error during training: {str(e)}")