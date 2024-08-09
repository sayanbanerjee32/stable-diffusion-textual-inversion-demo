import torch

# hue loss
def rgb_to_hsv(image):
    r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    maxc = torch.max(image, dim=1)[0]
    minc = torch.min(image, dim=1)[0]

    v = maxc
    s = (maxc - minc) / (maxc + 1e-10)
    deltac = maxc - minc

    # Initialize hue
    h = torch.zeros_like(maxc)

    mask = maxc == r
    h[mask] = ((g - b) / deltac)[mask] % 6

    mask = maxc == g
    h[mask] = ((b - r) / deltac)[mask] + 2

    mask = maxc == b
    h[mask] = ((r - g) / deltac)[mask] + 4

    h = h / 6  # Normalize to [0, 1]
    h[deltac == 0] = 0  # If no color difference, set hue to 0

    return torch.stack([h, s, v], dim=1)


def hue_loss(images, target_hue=0.5):
    # Convert the images to HSV color space
    hsv_images = rgb_to_hsv(images)

    # Extract the hue channel
    hue = hsv_images[:, 0, :, :]

    # Calculate the error as the mean absolute deviation from the target hue
    error = torch.abs(hue - target_hue).mean()

    return error
