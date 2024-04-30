import numpy as np
import torchmetrics
import torch

# 加载预测结果和验证集
pred_masks = np.load('/scratch/yz8297/DL_comp/dl2-master/Final_Pipeline/results/uint8_y_pred_masks_val_429.npy')  # 预测结果
#val_masks = np.load('/scratch/yz8297/DL_comp/Dataset_Student/val/video_01000/mask.npy')   # 加载验证集的标签，需要根据您的数据加载方式来完成
#val_mask = val_masks[-1]


jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
print("Data type:", pred_masks[0].dtype)

for i in range(1000,2000):
    video_id = '{:05d}'.format(i)
    val_masks = np.load('/scratch/yz8297/DL_comp/Dataset_Student/val/video_{}/mask.npy'.format(video_id))
    val_mask = val_masks[-1]
    pred_tensor = torch.tensor(pred_masks[i-1000])
    val_tensor = torch.tensor(val_mask)
    #iou = calculate_iou(pred_masks[i], val_mask)  # 计算 IoU
    #jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
    jaccard.update(pred_tensor, val_tensor)
    #iou_scores.append(iou)  # 将 IoU 添加到列表中
final_iou = jaccard.compute()
print(f"Final IoU on validation: {final_iou}")

