import numpy as np

# 加载.npy文件
pred_masks = np.load('/scratch/yz8297/DL_comp/dl2-master/Final_Pipeline/results/y_pred_masks_val_429.npy')

# 打印数组形状和数据类型
print("Shape:", pred_masks.shape)
print("Data type:", pred_masks.dtype)

# 将 int64 数组转换为 uint8 数组
uint8_pred_masks = pred_masks.astype(np.uint8)
save_path = '/scratch/yz8297/DL_comp/dl2-master/Final_Pipeline/results/uint8_y_pred_masks_val_429.npy'

print("Shape:", uint8_pred_masks.shape)
print("Data type:", uint8_pred_masks.dtype)

# 使用 np.save() 函数保存数组
np.save(save_path, uint8_pred_masks)

# 提示保存成功
print("uint8 array saved successfully at:", save_path)

