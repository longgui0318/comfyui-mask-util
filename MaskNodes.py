import os
import cv2
import torch
import shutil
import hashlib
import time
import folder_paths
import node_helpers
import numpy as np

from pathlib import Path
from comfy import model_management
from PIL import Image, ImageOps, ImageSequence, ImageFile
from .utils import *
from .resolution_utils import *

class ModelChangeDevice:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["default", "cpu", "cuda", "cuda:0", "cuda:1"],),
            },
            "optional": {
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "model":('MODEL', {}),
                "clip":('CLIP', {}),
                "vae":('VAE', {}),
                "pipeline":('PIPELINE', {}),
                "trigger_condition_image":("IMAGE",),
                "trigger_condition_mask": ("MASK",),
                "trigger_condition_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info", )
    FUNCTION = "changeDevice"

    CATEGORY = "image"

    def changeDevice(self, device = "default",sam_model=None,grounding_dino_model=None,model=None,clip=None,vae=None,pipeline=None,trigger_condition_image=None,trigger_condition_mask=None,trigger_condition_latent=None):
        #获取当前时间
        current_time = time.time()
        if device == "default":
            seleted_device = model_management.get_torch_device()
        else:
            seleted_device = torch.device(device)
        if sam_model is not None:
            if sam_model.device != seleted_device:
                sam_model.to(seleted_device);
        if grounding_dino_model is not None:
            if grounding_dino_model.device != seleted_device:
                grounding_dino_model.to(seleted_device);
        if pipeline is not None:
            if pipeline.device != seleted_device:
                pipeline.to(seleted_device);
        if model is not None:
            if model.current_device != seleted_device:
                model.model.to(seleted_device)
                model.current_device = seleted_device
        if clip is not None:
            if clip.patcher.current_device != seleted_device:
                clip.patcher.model.to(seleted_device)
                clip.patcher.current_device = seleted_device
        if vae is not None:
            if vae.patcher.current_device != seleted_device:
                vae.patcher.model.to(seleted_device)
                vae.patcher.current_device = seleted_device
                vae.device = seleted_device
        use_time = (time.time() - current_time)/1000   
        return (f"耗时:{use_time}s",)


class OutputImageToInput:
    @classmethod
    def INPUT_TYPES(s):
        output_dir = folder_paths.get_output_directory()
        files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        return {"required":
                    {"image": (sorted(files),)},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("STRING",)
    FUNCTION = "copy_image"
    
    def take_un_existed_name(self, image,no=0):
        if no>0:
            image_name = Path(image)
            real_image = image_name.stem+f"_{str(no)}"+image_name.suffix
        else:
            real_image = image
        target_image_path = os.path.join(folder_paths.get_input_directory(), real_image)
        if os.path.exists(target_image_path):
            #需要在后缀前加上_copy
            return self.take_un_existed_name(image,no+1)
        return (target_image_path,real_image)
        
    
    def copy_image(self, image):
        image_path = os.path.join(folder_paths.get_output_directory(), image)
        target_image_path,target_image = self.take_un_existed_name(image)
        shutil.copy(image_path, target_image_path)
        return (target_image,)


class LoadImageWithName:
    # copy from comfyui
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"image": ("STRING", {
                    "dynamicPrompts": False,
                    "multiline": False,
                    "default": ""
                }), }
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = os.path.join(folder_paths.get_input_directory(), image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

class ImageChangeDType:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dtype": (["default", "float32", "float16", "bfloat16"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    FUNCTION = "changeDType"

    CATEGORY = "image"

    def changeDType(self, image, dtype = "default"):
        if dtype == "float16":
            seleted_type = torch.float16
        elif dtype == "bfloat16":
            seleted_type = torch.bfloat16
        else:
            seleted_type = torch.float32
        image = image.to(dtype=seleted_type)
        return (image,)


class MaskChangeDType:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "dtype": (["default", "float32", "float16", "bfloat16"],),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask", )
    FUNCTION = "changeDType"

    CATEGORY = "mask"

    def changeDType(self, mask, dtype = "default"):
        if dtype == "float16":
            seleted_type = torch.float16
        elif dtype == "bfloat16":
            seleted_type = torch.bfloat16
        else:
            seleted_type = torch.float32
        mask = mask.to(dtype=seleted_type)
        return (mask,)

class ImageChangeDevice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["default", "cpu", "cuda", "cuda:0", "cuda:1"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    FUNCTION = "changeDevice"

    CATEGORY = "image"

    def changeDevice(self, image, device = "default"):

        if device == "default":
            seleted_device = model_management.get_torch_device()
        else:
            seleted_device = torch.device(device)
        if image.device != seleted_device:
            image = image.to(seleted_device) 
        return (image,)


class MaskChangeDevice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "device": (["default", "cpu", "cuda", "cuda:0", "cuda:1"],),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask", )
    FUNCTION = "changeDevice"

    CATEGORY = "mask"

    def changeDevice(self, mask, device = "default"):

        if device == "default":
            seleted_device = model_management.get_torch_device()
        else:
            seleted_device = torch.device(device)
        if mask.device != seleted_device:
            mask = mask.to(seleted_device) 
        return (mask,)

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


class ImageAdaptiveCrop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "is_xl": (["yes", "no"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "INT", "INT",
                    "INT", "INT", "INT", "INT", "INT",)
    # expand预期计算出来的外扩值，通过这个在生成图像中扣除新区域
    # crop是目标区域从原图从哪个位置抠出来的，需要把根据expand找出来的区域填充到crop区域
    RETURN_NAMES = ("adaptive_croped_image", "resolution_w", "resolution_h" "image_multiple", "expand_top", "expand_buttom", "expand_left",
                    "expand_right", "crop_area_x", "crop_area_y", "crop_area_width", "crop_area_height",)

    FUNCTION = "adaptiveCrop"

    CATEGORY = "image"

    def adaptiveCrop(self, image, crop_mask, expand, is_xl):
        # 将pythorch的tensor转换为opencv需要的图像格式
        crop_mask_cv2 = crop_mask.cpu().squeeze(0).numpy()
        crop_mask_cv2 = (crop_mask_cv2 * 255).astype(np.uint8)
        # 寻找轮廓
        contours, _ = cv2.findContours(
            crop_mask_cv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or len(contours) == 0:
            return (image, 1, 0, 0, 0, 0, 0, 0, 0, 0,)
        image_cv2 = image.cpu().squeeze(0).numpy()
        image_cv2 = (image_cv2 * 255).astype(np.uint8)
        image_w = image_cv2.shape[1]
        image_h = image_cv2.shape[0]
        image_left = 0
        image_top = 0
        image_right = image_w
        image_buttom = image_h

        all_contours = np.vstack(contours)
        all_contour_x, all_contour_y, all_contour_w, all_contour_h = cv2.boundingRect(
            all_contours)
        box_left = all_contour_x
        box_top = all_contour_y
        box_right = all_contour_x + all_contour_w
        box_bottom = all_contour_y + all_contour_h
        box_left = max(image_left, box_left - expand)
        box_top = max(image_top, box_top - expand)
        box_right = min(image_right, box_right + expand)
        box_bottom = min(image_buttom, box_bottom + expand)
        box_w = box_right - box_left
        box_h = box_bottom - box_top
        if is_xl == "yes":
            resolution_w, resolution_h, resolution_scale = resolution_auto_cel_from_sdxl(
                box_w, box_h)
        else:
            resolution_w, resolution_h, resolution_scale = resolution_auto_cel_from_sd15(
                box_w, box_h)
        if resolution_scale == 0:
            return (image, 1, 0, 0, 0, 0, 0, 0, 0, 0,)
        resolution_wt = resolution_w / resolution_scale
        resolution_ht = resolution_h / resolution_scale
        if resolution_wt > image_w:  # 预期的画图比原图还大
            rw_offset = (resolution_wt - image_w)//2
            # 因为生成的比原图大，所以不需要再使用原图抠出来了
            crop_area_x = 0
            crop_area_width = image_w
            # 需要外扩的宽度
            expand_left = rw_offset
            expand_right = (resolution_wt - image_w) - rw_offset
        elif resolution_wt > box_w:  # 预期的画图比原图小但比期望的空间大
            rw_offset = (image_w - resolution_wt)//2
            crop_area_x = max(0, box_left-rw_offset)
            crop_area_width = resolution_wt
            # 因为比图像小，不需要外扩
            expand_left = 0
            expand_right = 0
        else:
            crop_area_x = box_left
            crop_area_width = box_w
            # 因为比图像小，不需要外扩
            expand_left = 0
            expand_right = 0
        if resolution_ht > image_h:  # 预期的画图比原图还大
            rh_offset = (resolution_ht - image_h)//2
            # 因为生成的比原图大，所以不需要再使用原图抠出来了
            crop_area_y = 0
            crop_area_height = image_h
            # 需要外扩的高度
            expand_top = rh_offset
            expand_buttom = (resolution_ht - image_h) - rh_offset
        elif resolution_ht > box_h:  # 预期的画图比原图小但比期望的空间大
            rh_offset = (image_h - resolution_ht)//2
            crop_area_y = max(0, box_top-rh_offset)
            crop_area_height = resolution_ht
            # 因为比图像小，不需要外扩
            expand_top = 0
            expand_buttom = 0
        else:
            crop_area_y = box_top
            crop_area_height = box_h
            # 因为比图像小，不需要外扩
            expand_top = 0
            expand_buttom = 0

        # 使用numpy切片裁剪图像
        if crop_area_y > 0:
            image_cv2 = image_cv2[crop_area_y:, :]
        if image_h-(crop_area_y+crop_area_height) > 0:
            image_cv2 = image_cv2[:-(image_h-(crop_area_y+crop_area_height)), :]
        if crop_area_x > 0:
            image_cv2 = image_cv2[:, crop_area_x:]
        if image_w-(crop_area_x+crop_area_width) > 0:
            image_cv2 = image_cv2[:, :-(image_w-(crop_area_x+crop_area_width))]
            
        if expand_top>0 or expand_buttom>0 or expand_left>0 or expand_top>0:
            # 使用cv2.copyMakeBorder函数扩展图像
            image_cv2 = cv2.copyMakeBorder(image_cv2, expand_top, expand_buttom, expand_left, expand_right, cv2.BORDER_REPLICATE)
            
        image_cv2 = np.array(image_cv2).astype(np.float32) / 255.0
        image_cv2 = torch.from_numpy(image_cv2)
        return (image_cv2, resolution_w, resolution_h, resolution_scale, expand_top, expand_buttom, expand_left, expand_right, crop_area_x, crop_area_y, crop_area_width, crop_area_height,)


class ImageResolutionLimitWith8K:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit": ("INT", {"default": 8192, "min": 512, "max": 8192, "step": 128}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    FUNCTION = "limit8K"

    CATEGORY = "image"

    def limit8K(self, image, limit=8192):
        if limit == 0:
            return (image,)
        if image.shape[1] <= limit and image.shape[2] <= limit:
            return (image,)
        # 获取图像的宽度和高度
        opencv_image = image.cpu().squeeze(0).numpy()
        opencv_image = (opencv_image * 255).astype(np.uint8)
        height = opencv_image.shape[0]
        width = opencv_image.shape[1]
        aspect_ratio = width / height
        # 计算新的宽和高，确保不超过max_length
        if width > height:
            new_width = limit
            new_height = int(limit / aspect_ratio)
        else:
            new_height = limit
            new_width = int(limit * aspect_ratio)

        # 使用cv2.resize进行缩小，同时保持宽高比
        resized_img = cv2.resize(opencv_image, (new_width, new_height))
        numpy_image = np.array(resized_img).astype(np.float32) / 255.0
        numpy_image = torch.from_numpy(numpy_image)
        return (numpy_image.unsqueeze(0),)


class ImageResolutionAdaptiveWithX:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["crop", "expand"],),
                "base": ("INT", {"default": 8, "min": 0, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    FUNCTION = "adaptive"

    CATEGORY = "image"

    def adaptive(self, image, mode="crop", base=8):
        if base == 0:
            return (image,)
        # 获取图像的宽度和高度
        opencv_image = image.cpu().squeeze(0).numpy()
        opencv_image = (opencv_image * 255).astype(np.uint8)
        height = opencv_image.shape[0]
        width = opencv_image.shape[1]
        height_offset = height % base
        width_offset = width % base
        if mode == "expand":
            height_offset = base - height_offset
            width_offset = base - width_offset
        else:
            height_offset = height_offset
            width_offset = width_offset
        if mode == "expand":
            top = height_offset//2
            bottom = height_offset - top
            left = width_offset//2
            right = width_offset - left
        else:
            top = height_offset//2
            bottom = height_offset - top
            left = width_offset//2
            right = width_offset - left
        if mode == "expand":
            # 使用cv2.copyMakeBorder函数扩展图像
            opencv_image = cv2.copyMakeBorder(
                opencv_image, top, bottom, left, right, cv2.BORDER_REPLICATE)
        else:
            # 使用numpy切片裁剪图像
            if top > 0:
                opencv_image = opencv_image[top:, :]
            if bottom > 0:
                opencv_image = opencv_image[:-bottom, :]
            if left > 0:
                opencv_image = opencv_image[:, left:]
            if right > 0:
                opencv_image = opencv_image[:, :-right]
        numpy_image = np.array(opencv_image).astype(np.float32) / 255.0
        numpy_image = torch.from_numpy(numpy_image)
        return (numpy_image.unsqueeze(0),)


class ImageCropWithMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "width": ("INT", {"default": 512, "min": 0, "max": 8096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 8096, "step": 1}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    FUNCTION = "adaptive"

    CATEGORY = "image"

    def adaptive(self, image,mask,width,height, mode="zoom",padding=0):
        # 将pythorch的tensor转换为opencv需要的图像格式
        mask_cv2 = mask.cpu().squeeze(0).numpy()
        mask_cv2 = (mask_cv2 * 255).astype(np.uint8)
        _, mask_cv2 = cv2.threshold(mask_cv2, 1, 255, cv2.THRESH_BINARY)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(mask_cv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or len(contours) == 0:
            return (image)
        image_cv2 = image.cpu().squeeze(0).numpy()
        image_cv2 = (image_cv2 * 255).astype(np.uint8)
        mask_bool = mask_cv2 > 0
        image_cv2[~mask_bool]=0

        all_contours = np.vstack(contours)
        all_contour_x, all_contour_y, all_contour_w, all_contour_h = cv2.boundingRect(all_contours)
        box_left = all_contour_x
        box_top = all_contour_y
        box_right = all_contour_x + all_contour_w
        box_bottom = all_contour_y + all_contour_h
        box_left = max(0, box_left - padding)
        box_top = max(0, box_top - padding)
        box_right = min(image_cv2.shape[1], box_right + padding)
        box_bottom = min(image_cv2.shape[0], box_bottom + padding)
        
        image_cv2 = image_cv2[box_top:box_bottom,box_left:box_right]
        
        
        scale_w,scale_h = resize_to(image_cv2.shape[1],image_cv2.shape[0],width,height)
        image_cv2 = cv2.resize(image_cv2, (scale_w,scale_h))
        offset_w = width - image_cv2.shape[1]
        offset_h = height - image_cv2.shape[0]
        offset_top = offset_h//2
        offset_bottom = offset_h - offset_top
        offset_left = offset_w//2
        offset_right = offset_w - offset_left
        image_cv2 = cv2.copyMakeBorder(image_cv2, offset_top, offset_bottom, offset_left, offset_right, cv2.BORDER_CONSTANT,0)
        image_cv2 = np.array(image_cv2).astype(np.float32) / 255.0
        image_cv2 = torch.from_numpy(image_cv2)
        return (image_cv2.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "Model Change Device": ModelChangeDevice,
    "Load Image With Name": LoadImageWithName,
    "Output Image To Input": OutputImageToInput,
    "Mask Change DType": MaskChangeDType,
    "Image Change DType": ImageChangeDType,
    "Mask Change Device": MaskChangeDevice,
    "Image Change Device": ImageChangeDevice,
    "Split Masks": SplitMasks,
    "Mask Selection Of Masks": MaskselectionOfMasks,
    "Image Adaptive Crop M&R": ImageAdaptiveCrop,
    "Image Adaptive Crop With Mask": ImageCropWithMask,
    "Image Resolution Limit With 8K": ImageResolutionLimitWith8K,
    "Image Resolution Adaptive With X": ImageResolutionAdaptiveWithX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Model Change Device": "Model Change Device",
    "Load Image With Name": "Load Image With Name",
    "Output Image To Input": "Output Image To Input",
    "Mask Change DType": "Mask Change DType",
    "Image Change DType": "Image Change DType",
    "Mask Change Device": "Mask Change Device",
    "Image Change Device": "Image Change Device",
    "Split Masks": "Split Masks",
    "Mask Selection Of Masks": "Mask Selection Of Masks",
    "Image Adaptive Crop M&R": "Image Adaptive Crop MASK&Resolution",
    "Image Adaptive Crop With Mask": "Image Adaptive Crop With Mask",
    "Image Resolution Limit With 8K": "Image Resolution Limit With 8K",
    "Image Resolution Adaptive With X": "Image Resolution Adaptive With X",
}
