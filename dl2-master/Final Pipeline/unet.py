import os
import torch
import numpy as np
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class SegmentationDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集。
        参数:
            root_dir (str): 包含所有视频文件夹的根目录。
            transform (callable, optional): 一个可选的转换函数或复合转换，将应用于加载的图像。
        """
        self.transform = transform
        self.images = []
        # 遍历根目录下的所有文件夹
        for dir_name in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, dir_name)
            # 确保它是一个目录
            if os.path.isdir(dir_path):
                imgs = os.listdir(dir_path)
                self.images.extend([os.path.join(dir_path, img) for img in imgs if img.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')  # 确保图像为RGB格式
        if self.transform:
            img = self.transform(img)  # 应用transform
        img = np.array(img)  # 转换图像为numpy数组
        img = torch.from_numpy(img.transpose((2, 0, 1)))  # 转换为CHW格式，适应PyTorch
        return img

class encoding_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoding_block, self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*model)

    def forward(self, x):
        return self.conv(x)


class unet_model(nn.Module):
    def __init__(self, out_channels=49, features=[64, 128, 256, 512]):
        super(unet_model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = encoding_block(3, features[0])
        self.conv2 = encoding_block(features[0], features[1])
        self.conv3 = encoding_block(features[1], features[2])
        self.conv4 = encoding_block(features[2], features[3])
        self.conv5 = encoding_block(features[3] * 2, features[3])
        self.conv6 = encoding_block(features[3], features[2])
        self.conv7 = encoding_block(features[2], features[1])
        self.conv8 = encoding_block(features[1], features[0])
        self.tconv1 = nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.bottleneck = encoding_block(features[3], features[3] * 2)
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x


def UNET_Module(args):
    model2_path = args.model2_path


    val_dataset = SegmentationDataSet(args, None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    DEVICE = torch.device('cuda:{}'.format(0))

    model = unet_model().to(DEVICE)

    loaded_unet_model = torch.load(model2_path).state_dict()
    model.load_state_dict(loaded_unet_model)

    model.eval()

    masks_pred_list = []

    with torch.no_grad():
        for x in tqdm(val_dataloader):
            #print("x.shape:",x.shape)
            #print("x.type:",x.type)

            x = x.type(torch.cuda.FloatTensor).to(DEVICE)

            softmax = nn.Softmax(dim=1)

            preds = torch.argmax(softmax(model(x)), axis=1)

            masks_pred_list.append(preds)


    torch_y_pred_masks=torch.cat(masks_pred_list,dim=0)
    numpy_y_pred_masks=torch_y_pred_masks.to('cpu').numpy()

    print("After segmentation shape", numpy_y_pred_masks.shape)

    np.save(args.res_dir+'/unlabled.npy',numpy_y_pred_masks)
    print("segmentation done successfully")

