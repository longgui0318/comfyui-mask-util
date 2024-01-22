import torch
import numpy as np
import cv2
 
 
class SplitMasks:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ignore_threshold": ("INT",),
            },
        }

    RETURN_TYPES = ("images","masks",)
    FUNCTION = "separate"

    CATEGORY = "mask"

    def separate(self, image,ignore_threshold=100):
        # 将pythorch的tensor转换为numpy的array
        numpy_image = image.numpy()
        # 将numpy的array转换为opencv的image
        opencv_gray_image = cv2.cvtColor(numpy_image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
        # 阈值处理，将图像二值化
        _, binary_mask = cv2.threshold(opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        # 寻找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个空白图像，用于绘制分割结果
        segmented_masks = []
        for contour in contours:
            # 计算轮廓的面积
            area = cv2.contourArea(contour)
            # 如果面积小于忽略阈值，则跳过
            if area > ignore_threshold:
                continue
            
            # 创建一个空白图像，用于绘制分割结果
            segmented_mask = np.zeros_like(opencv_gray_image)
            # 绘制轮廓
            cv2.drawContours(segmented_mask, [contour], 0, 255, thickness=cv2.FILLED)
            # 将当前分割好的遮罩添加到列表中
            segmented_masks.append(segmented_mask)
            
        output_images = []
        output_masks = []

        for segmented_mask in segmented_masks:
            # 将 NumPy 数组转换为 torch
            mask =  torch.from_numpy(segmented_mask)

            # 添加维度
            mask = mask.unsqueeze(0)

            # 创建一个空白图像（可以根据需要进行初始化）
            image = torch.zeros_like(mask)

            # 将图像和遮罩添加到输出列表
            output_images.append(image)
            output_masks.append(mask)
        
        return (output_images,output_masks,)
 

NODE_CLASS_MAPPINGS = {
     "Split Masks": SplitMasks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Split Masks": "Split Masks",
}