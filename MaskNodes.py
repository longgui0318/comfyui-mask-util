import torch
import numpy as np
import cv2
from .utils import *


class SplitMasks:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks", )
    FUNCTION = "separate"

    CATEGORY = "mask"

    def separate(self, mask, ignore_threshold=100):
        # 将pythorch的tensor转换为opencv需要的图像格式
        opencv_gray_image = tensorMask2cv2img(mask)
        # 阈值处理，将图像二值化
        _, binary_mask = cv2.threshold(
            opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        # 寻找轮廓
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 创建一个空白图像，用于绘制分割结果
        segmented_masks = []
        for contour in contours:
            # 计算轮廓的面积
            area = cv2.contourArea(contour)
            # 如果面积小于忽略阈值，则跳过
            if area < ignore_threshold:
                continue

            # 创建一个空白图像，用于绘制分割结果
            segmented_mask = np.zeros_like(binary_mask)
            # 绘制轮廓
            cv2.drawContours(segmented_mask, [
                             contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            # 将当前分割好的遮罩添加到列表中
            segmented_masks.append(segmented_mask)
        output_masks = []

        for segmented_mask in segmented_masks:
            numpy_mask = np.array(segmented_mask).astype(np.float32) / 255.0
            i_mask = torch.from_numpy(numpy_mask)
            # 将图像和遮罩添加到输出列表
            output_masks.append(i_mask.unsqueeze(0))
        return (output_masks,)


class MaskselectionOfMasks:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "index": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask", )
    FUNCTION = "selection"

    CATEGORY = "mask"

    def selection(self, mask, index=0):
        # 判断 mask 是否是数组
        if isinstance(mask, list):
            # 如果是数组，则直接返回数组中的第一个元素
            result = mask[index]
        else:
            # 如果不是数组，则直接返回 mask
            result = mask
        if result is None:
            result = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (result,)


class MaskRegionInfo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("mask", "x", "y", "w", "h",)

    FUNCTION = "takeInfo"

    CATEGORY = "mask"

    def takeInfo(self, mask):
        # 将pythorch的tensor转换为opencv需要的图像格式
        opencv_gray_image = tensorMask2cv2img(mask)
        # 阈值处理，将图像二值化
        _, binary_mask = cv2.threshold(
            opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        # 寻找轮廓
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or len(contours) == 0:
            return (mask, 0, 0, 0, 0,)
        all_contours = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_contours)
        return (mask, x, y, w, h,)


NODE_CLASS_MAPPINGS = {
    "Split Masks": SplitMasks,
    "Mask Selection Of Masks": MaskselectionOfMasks,
    "Mask Region Info": MaskRegionInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Split Masks": "Split Masks",
    "Mask Selection Of Masks": "Mask Selection Of Masks",
    "Mask Region Info": "Mask Region Info",
}
