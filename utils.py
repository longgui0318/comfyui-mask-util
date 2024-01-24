import numpy as np
import cv2
import torch
from PIL import Image
 
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    # Check if the image_tensor is a list of tensors
    if isinstance(image_tensor, list):
        # Initialize an empty list to store the converted images
        image_numpy = []
        # Loop through each tensor in the list
        for i in range(len(image_tensor)):
            # Recursively call the tensor2im function on each tensor and append the result to the list
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        # Return the list of converted images
        return image_numpy
    # If the image_tensor is not a list, convert it to a NumPy array on the CPU with float data type
    image_numpy = image_tensor.cpu().float().numpy()

    # Check if the normalize parameter is True
    if normalize:
        # This will scale the pixel values from [-1, 1] to [0, 255]
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        # This will scale the pixel values from [0, 1] to [0, 255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0

        # Clip the pixel values to the range [0, 255] to avoid overflow or underflow
    image_numpy = np.clip(image_numpy, 0, 255)
    # Check if the array has one or more than three channels
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        # If so, select only the first channel and discard the rest
        # This will convert the array to grayscale
        image_numpy = image_numpy[:, :, 0]
    # Return the array with the specified data type (default is unsigned 8-bit integer)
    return image_numpy.astype(imtype)

 

def tensor2cv2img(tensor) -> np.ndarray:
    # Move the tensor to the CPU if needed,and revmoe the batch dimension
    tensor = tensor.cpu().squeeze(0)
    array = tensor.numpy()
    array = (array * 255).astype(np.uint8)
    # Convert the color space from RGB to GRAY
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

def tensorMask2cv2img(tensor) -> np.ndarray:
    # Move the tensor to the CPU if needed,and revmoe the batch dimension
    tensor = tensor.cpu().squeeze(0)
    array = tensor.numpy()
    array = (array * 255).astype(np.uint8)
    return array



def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
