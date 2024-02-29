recommended_sd15_resolutions = [
    (512, 512, 1),
    (512, 768, 0.666666666),
    (768, 768, 1),
]


recommended_sdxl_resolutions = [
    (256, 512, 0.5),
    (512, 512, 1.0),
    (512, 768, 0.666666666),
    (512, 1024, 0.5),
    (512, 1856, 0.275862069),
    (512, 1920, 0.266666667),
    (512, 1984, 0.258064516),
    (512, 2048, 0.25),
    (576, 1664, 0.346153846),
    (576, 1728, 0.333333333),
    (576, 1792, 0.321428571),
    (640, 1536, 0.416666667),
    (640, 1600, 0.346153846),
    (704, 1344, 0.523809524),
    (704, 1408, 0.5),
    (704, 1472, 0.47826087),
    (768, 768, 1.0),
    (768, 1280, 0.6),
    (768, 1344, 0.571428571),
    (832, 1152, 0.722222222),
    (832, 1216, 0.684210526),
    (896, 1088, 0.823529412),
    (896, 1152, 0.777777778),
    (960, 1024, 0.9375),
    (960, 1088, 0.882352941),
    (1024, 1024, 1.0),
]


def resolution_auto_cel_from_sd15(w, h):
    return resolution_auto_cel(w,h,recommended_sd15_resolutions)
    
def resolution_auto_cel_from_sdxl(w, h):
    return resolution_auto_cel(w,h,recommended_sdxl_resolutions)



def resolution_auto_cel(w, h, base_resolutions):
    # 宽高调整，小在前
    is_swap = False
    if w > h:
        is_swap = True
        w, h = h, w
    scale_resolutions = []
    for resolution in base_resolutions:
        # 将w缩放到resolution[0]的分辨率
        scale = resolution[0]/w
        scale_resolutions.append(
            (resolution[0], resolution[1], resolution[2], scale))
    # 找出最接近的分辨率
    similar_item = scale_resolutions[0]
    for resolution in scale_resolutions:
        if abs(resolution[1] / resolution[3] - h) < abs(similar_item[1] / similar_item[3] - h):
            similar_item = resolution
    # 因为目标按照期望比例处理时，无法达到期望的高度，所以需要重新计算scale，转而使用高度约束来进行缩放
    if (similar_item[1] / similar_item[3]) < h:
        similar_item = (similar_item[0], similar_item[1],
                        similar_item[2], similar_item[1]/h)
    # 找到最佳的缩放比例，要求该缩放比例找出来的值肯定都是整数，同时比需要的要大
    bast_scale = find_bast_scale(
        similar_item[0], similar_item[1], similar_item[3])
    if bast_scale == 0:
        raise ValueError("Unable to find the right scaling.")
    if is_swap:
        return (similar_item[1], similar_item[0], bast_scale)
    else:
        return (similar_item[0], similar_item[1], bast_scale)


def find_bast_scale(x, y, current_scale):
    bast_scale = int(current_scale*100)
    for _ in range(99):
        if bast_scale == 0:
            return 0
        xS = x/(bast_scale/100.0)
        yS = y/(bast_scale/100.0)
        if xS.is_integer() and yS.is_integer():
            return bast_scale/100.0
        bast_scale += -1
    return 0