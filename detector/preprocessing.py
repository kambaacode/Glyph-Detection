import cv2
import numpy as np


def correct_illumination(img, kernel_size=151):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    blur = np.where(blur < 1, 1, blur).astype(np.float32)
    ratio = gray / blur
    result = img.astype(np.float32) * ratio[..., np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def denoise_image(img, strength=7):
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def adjust_gamma(img, gamma=1.0):
    inv_gamma = 1.0 / max(gamma, 0.01)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def enhance_contrast_clahe(img, clip_limit=3.0, tile_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def morphological_tophat(img, kernel_size=31):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)


def unsharp_mask(img, strength=1.5, radius=5, threshold=10):
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def increase_saturation(img, scale=1.3):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def auto_contrast_stretch(img, low_percent=2, high_percent=98):
    channels = cv2.split(img)
    stretched = []
    for ch in channels:
        low_val = np.percentile(ch, low_percent)
        high_val = np.percentile(ch, high_percent)
        stretched.append(cv2.normalize(ch, None, alpha=0, beta=255,
                                       norm_type=cv2.NORM_MINMAX))
    return cv2.merge(stretched)


def preprocess_for_segmentation(img):
    corrected = correct_illumination(img, kernel_size=151)
    denoised = denoise_image(corrected, strength=5)
    contrast = enhance_contrast_clahe(denoised, clip_limit=3.0, tile_size=(8, 8))
    adjusted = adjust_gamma(contrast, gamma=1.1)
    sharp = unsharp_mask(adjusted, strength=1.2, radius=3, threshold=10)
    return sharp


def preprocess_for_classification(img):
    corrected = correct_illumination(img, kernel_size=51)
    enhanced = enhance_contrast_clahe(corrected, clip_limit=1.5, tile_size=(8, 8))
    return unsharp_mask(enhanced, strength=0.6, radius=1, threshold=3)
