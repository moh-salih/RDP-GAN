
import os
from datetime import date

import torch
import utils
from torchvision.utils import save_image
import pandas as pd
import matplotlib.pyplot as plt


# Initial settings (Strongly recommended to be changed)
num_epochs=10
model_save_interval=10
data_save_interval=10
show_log_interval=1
privacy_mode = 'no_privacy'
dataset_name='unknown'



# Helper functions
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
today = date.today().strftime("%d_%m_%Y")

def check_and_mkdir_if_necessary(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


def log(epoch, d_loss, g_loss, real_scores, fake_scores): 
    if epoch % show_log_interval != 0: return 
    print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
            'D real: {:.6f}, D fake: {:.6f}'.format(
                epoch, num_epochs, d_loss.item(), g_loss.item(),
                real_scores.data.mean(), fake_scores.data.mean()))


def save_model(epoch, generator, sigma=0, name=''):
    if epoch % model_save_interval != 0: return 

    model_path =  os.path.join(ROOT_DIR, 'data', 'working', today, dataset_name, 'models', privacy_mode)
    check_and_mkdir_if_necessary(model_path)

    model_name = f'epoch_{epoch}_sigma_{sigma}.pth'
    torch.save(generator.state_dict(), f'{model_path}/{model_name}')


def save_csv(epoch, data, sigma=0, file_name=''):
    if epoch % data_save_interval != 0: return
    fake_data_df = pd.DataFrame(data)

    data_path =  os.path.join(ROOT_DIR, 'data', 'working', today, dataset_name, 'data', privacy_mode)
    check_and_mkdir_if_necessary(data_path)
    file_name = f'epoch_{epoch}_sigma_{sigma}_{file_name}.csv'

    fake_data_df.to_csv(f'{data_path}/{file_name}', index=False, header=False)


def save_images(epoch, image, sigma):
    if epoch % data_save_interval != 0: return

    image = to_img(image.data)

    image_path =  os.path.join(ROOT_DIR, 'data', 'working', today, dataset_name, 'images', privacy_mode)
    check_and_mkdir_if_necessary(image_path)
    image_name = f'epoch_{epoch}_sigma_{sigma}.png' 
    
    save_image(image, f'{image_path}/{image_name}')


def losses_over_epoches(g_losses, d_losses,  epoch, sigma=0, x_label='Epoch', y_label='Loss', ):
    plt.clf()

    plt.plot(range(1, len(g_losses) + 1), g_losses, label='Generator Loss', color='blue')
    plt.plot(range(1, len(d_losses) + 1), d_losses, label='Discriminator Loss', color='red')


    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if privacy_mode == 'no_privacy':
        title='Generator and Discriminator Losses'
    else:
        title=f'Generator and Discriminator Losses with sigma = {sigma}'
    
    plt.title(title)
    
    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)
    
    # Show plot
    # plt.show()

    path =  os.path.join(ROOT_DIR, 'data', 'working', today, dataset_name, 'visualization', privacy_mode)
    check_and_mkdir_if_necessary(path)
    image_name = f'epoch_{epoch}_sigma_{sigma}.png'
    plt.savefig(f'{path}/{image_name}')