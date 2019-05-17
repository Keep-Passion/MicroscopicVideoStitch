import os
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from nets.ssim_sf_loss import SSIM_SF_Loss
from nets.mufoc_material_dataset import MufocMaterialDataset
from nets.models.unet_model import UNet
from nets.nets_utility import *


# parameter for net
experiment_name = 'sf_ssim_no_aug'
gpu_device = "cuda:0"
learning_rate = 1e-2
momentum = 0.9
epochs = 10
batch_size = 10
display_step = 100
shuffle = True


# address
project_addrsss = os.getcwd()
train_dir = os.path.join(os.path.join(os.path.join(project_addrsss,"datasets"),"used_for_nets"), "train")
val_dir = os.path.join(os.path.join(os.path.join(project_addrsss,"datasets"),"used_for_nets"), "val")
log_address = os.path.join(os.path.join(os.path.join(project_addrsss, "nets"), "train_record"), experiment_name + "_log_file.txt")
is_out_log_file = True
parameter_address = os.path.join(os.path.join(project_addrsss, "nets"), "parameters")


# datasets
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.3976813856328417], [0.05057423681276217]),
])
image_datasets = {}
image_datasets['train'] = MufocMaterialDataset(
    train_dir, transform=data_transforms, need_augment=False)
image_datasets['val'] = MufocMaterialDataset(
    val_dir, transform=data_transforms, need_augment=False)
dataloders = {}
dataloders['train'] = DataLoader(
    image_datasets['train'],
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=1)
dataloders['val'] = DataLoader(
    image_datasets['val'],
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=1)
datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print_and_log("datasets size: {}".format(datasets_sizes), is_out_log_file, log_address)

# models
training_setup_seed(1)  # setup seed for all parameters, numpy.random, random, pytorch
model = UNet(n_channels=1, n_classes=1, device=gpu_device)
model.to(gpu_device)
criterion = SSIM_SF_Loss().to(gpu_device)
# optimizer = optim.RMSprop(model.parameters(), learning_rate, momentum)
optimizer = optim.Adam(model.parameters(), learning_rate)


def val():
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloders['val']):
            # get the inputs
            patch1, patch2, patch_sf_y = data['last'].to(
                gpu_device), data['next'].to(gpu_device), data['sf'].to(gpu_device)
            # 先将网络中的所有梯度置0
            optimizer.zero_grad()
            patch1_lum = patch1[:, 0:1]
            patch2_lum = patch2[:, 0:1]
            # 网络的前向传播
            y_f = model.forward(patch1_lum, patch2_lum)
            loss = criterion(y_1=patch1_lum, y_2=patch2_lum, y_sf=patch_sf_y, y_f=y_f)

            # 记录当前batch_size的loss以及数据对应的分类准确数量
            running_loss += loss.item()

    epoch_loss_val = running_loss / datasets_sizes['val']
    return epoch_loss_val


def train(epoch):
    model.train()  # 把module设成training模式，对Dropout和BatchNorm有影响
    adjust_learning_rate(optimizer, learning_rate, epoch)
    print_and_log('Train Epoch {}/{}:'.format(epoch + 1, epochs), is_out_log_file, log_address)
    running_loss = 0.0
    # Iterate over data.
    for i, data in enumerate(dataloders['train']):
        # get the inputs
        patch1, patch2, patch_sf_y = data['last'].to(
            gpu_device), data['next'].to(gpu_device), data['sf'].to(gpu_device)

        patch1_lum = patch1[:, 0:1]
        patch2_lum = patch2[:, 0:1]
        #         print(patch1_lum.dtype)
        #         print(np.unique(patch1_lum.cpu().numpy()))
        #         print(patch2_lum.dtype)
        #         print(np.unique(patch2_lum.cpu().numpy()))
        #         input()
        # 网络的前向传播
        y_f = model.forward(patch1_lum, patch2_lum)
        loss = criterion(y_1=patch1_lum, y_2=patch2_lum, y_sf=patch_sf_y, y_f=y_f)

        # 记录当前batch_size的loss以及数据对应的分类准确数量
        running_loss += loss.item()

        if i % display_step == 0:
            print_and_log('\t{} {}-{}: Loss: {:.4f}'.format('train', epoch + 1, i, loss.item() / batch_size),
                          is_out_log_file, log_address)

        # 先将网络中的所有梯度置0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss_train = running_loss / datasets_sizes['train']
    return epoch_loss_train


def main():
    min_loss = 100000000.0
    loss_train = []  # 训练集loss
    loss_val = []  # 验证集loss
    since = time.time()
    for epoch in range(epochs):
        epoch_loss_train = train(epoch)
        loss_train.append(epoch_loss_train)
        epoch_loss_val = val()
        loss_val.append(epoch_loss_val)
        print_and_log('\ttrain Loss: {:.6f}'.format(epoch_loss_train), is_out_log_file, log_address)
        print_and_log('\tvalidation Loss: {:.6f}'.format(epoch_loss_val), is_out_log_file, log_address)

        # deep copy the model
        if epoch_loss_val < min_loss:
            min_loss = epoch_loss_val
            best_model_wts = model.state_dict()
            print_and_log("Updating", is_out_log_file, log_address)
            torch.save(best_model_wts,
                       os.path.join(parameter_address, experiment_name + '.pkl'))
        plot_loss(experiment_name, epoch, loss_train, loss_val)
        model_wts = model.state_dict()
        torch.save(model_wts,
                   os.path.join(parameter_address, experiment_name + "_" + str(epoch) + '.pkl'))
        time_elapsed = time.time() - since
        print_and_log('Time passed {:.0f}h {:.0f}m {:.0f}s'.
                      format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60), is_out_log_file,
                      log_address)
        print_and_log('-' * 20, is_out_log_file, log_address)
    print_and_log("train loss: {}".format(loss_train), is_out_log_file, log_address)
    print_and_log("val loss: {}".format(loss_val), is_out_log_file, log_address)
    print_and_log("min val loss: {}".format(min(loss_val)), is_out_log_file, log_address)