import os
import gzip
import random
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torch.utils.data import random_split, Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_num(s):
    return int(s.split('_')[1])

def extract_image_num(s):
    return int(s.split('_')[1].split('.')[0])

class MovingObjectDataSet(data.Dataset):
    def __init__(self, root, is_train=True, n_frames_input=11, n_frames_output=11, transform=None):
        super(MovingObjectDataSet, self).__init__()

        self.videos = []
        unlabelled_dirs = os.listdir(root)
        unlabelled_dirs = sorted(unlabelled_dirs, key=extract_num)

        for video in unlabelled_dirs:
            self.videos.extend([root + '/' + video + '/'])
        
        self.length = len(self.videos)

        self.is_train = is_train

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.mean = 0
        self.std = 1

    def __getitem__(self, index):
        video_folder_path = self.videos[index]
        video_files = os.listdir(video_folder_path)
        video_files = sorted(video_files, key=extract_image_num)

        # 确保文件列表非空
        if not video_files:
            return   # 或其他错误处理逻辑

        imgs = []
        last_valid_img = None
        for file_name in video_files:
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(video_folder_path, file_name)
                    img = Image.open(img_path)
                    last_valid_img = np.array(img)
                    imgs.append(last_valid_img)
                except Exception as e:
                    print(f"Error opening image {img_path}: {e}")
                    if last_valid_img is not None:
                        imgs.append(last_valid_img)  # 使用上一有效帧作为回退

        if not imgs:  # 如果没有成功加载任何图片
            return 

        # 转换逻辑...
        past_clips = imgs[:self.n_frames_input]
        future_clips = imgs[-self.n_frames_output:]

        past_clips = torch.stack([torch.from_numpy(np.transpose(clip, (2, 0, 1))) for clip in past_clips])
        future_clips = torch.stack([torch.from_numpy(np.transpose(clip, (2, 0, 1))) for clip in future_clips])

        return past_clips.float(), future_clips.float()

    def __len__(self):
        return self.length

def load_moving_object(batch_size, val_batch_size, data_root, num_workers):
    whole_data = MovingObjectDataSet(root=data_root, is_train=True, n_frames_input=11, n_frames_output=11)

    # 直接获取数据集总长度
    total_length = len(whole_data)
    
    # 手动设置验证集和测试集的大小
    val_size = int(0.05 * total_length)
    test_size = int(0.05 * total_length)  # 确保总和不超过数据集大小
    
    # 训练集大小自动调整
    train_size = total_length - val_size - test_size
    
    # 使用 random_split 进行分割
    train_data, val_data, test_data = random_split(whole_data, [train_size, val_size, test_size],
                                                   generator=torch.Generator().manual_seed(2021))

    # 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    mean, std = 0, 1  # 定义你的实际均值和标准差
    return train_loader, val_loader, test_loader, mean, std

