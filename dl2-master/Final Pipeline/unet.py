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
        self.transform = transform
        self.images = []
        # 遍历根目录下的所有子目录
        for dir_name in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                imgs = os.listdir(dir_path)
                self.images.extend([os.path.join(dir_path, img) for img in imgs if img.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB
        if self.transform:
            img = self.transform(img)  # Apply transformations
        img = np.array(img)  # Convert image to numpy array
        img = torch.from_numpy(img.transpose((2, 0, 1)))  # Convert to CHW format
        img = img.float()  # Convert to float
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
from PIL import Image

def save_images(numpy_array, output_dir, num_images_per_folder=22):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx in range(0, len(numpy_array), num_images_per_folder):
        folder_name = os.path.join(output_dir, f"video_{idx // num_images_per_folder:05d}")
        os.makedirs(folder_name, exist_ok=True)
        for j, image_array in enumerate(numpy_array[idx:idx + num_images_per_folder]):
            img = Image.fromarray((image_array * 255).astype(np.uint8))  # Assuming the output is normalized
            img.save(os.path.join(folder_name, f"frame_{j:05d}.png"))

def UNET_Module(args):
    model2_path = args.model2_path
    root_video_dir = args.data_root
    val_dataset = SegmentationDataSet(root_video_dir, None)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 直接加载整个模型
    model = torch.load(model2_path, map_location=device)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()
    masks_pred_list = []

    with torch.no_grad():
        for x in tqdm(val_dataloader):
            x = x.to(device).float()  # Ensure x is float before passing to model
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)), axis=1)
            masks_pred_list.append(preds)

    torch_y_pred_masks = torch.cat(masks_pred_list, dim=0)
    numpy_y_pred_masks = torch_y_pred_masks.cpu().numpy()

    print("After segmentation shape:", numpy_y_pred_masks.shape)

    np.save(os.path.join(args.res_dir, 'unlabeled.npy'), numpy_y_pred_masks)
    save_images(numpy_y_pred_masks, os.path.join(args.res_dir, 'unlabeled'))
    print("Segmentation done successfully")


