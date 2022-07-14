import lpips
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
import os
from tqdm import tqdm


def get_prefix(idx):
    return '0' * (4 - np.floor(np.log10(idx)).astype(int))


def img_to_tensor(im):
    return torch.tensor(np.array(im.convert('RGB')).astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0) * 2 - 1


def tensor_to_image(t):
    return Image.fromarray(np.array(((t.squeeze().permute(1, 2, 0) + 1) / 2).clip(0, 1) * 255).astype(np.uint8))


def diffuse(img, t):
    z = torch.randn_like(img)
    noisy_image = np.sqrt(alphas_bar[t]) * img + np.sqrt(one_minus_alphas_bar[t]) * z
    return tensor_to_image(noisy_image)


def diffuse_t(img, t):
    z = torch.randn_like(img)
    noisy_image = np.sqrt(alphas_bar[t]) * img + np.sqrt(one_minus_alphas_bar[t]) * z
    return noisy_image


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

diff_steps = 1000
betas = np.linspace(0.0001, 0.02, diff_steps)
alphas = 1 - betas
alphas_bar = np.array([np.prod(alphas[:i]) for i in range(1, len(alphas) + 1)])
one_minus_alphas_bar = 1 - alphas_bar
plt.plot(betas, label='beta')
plt.plot(alphas, label='alpha')
plt.plot(alphas_bar, label='alpha_bar')
plt.plot(one_minus_alphas_bar, label='1 - alpha_bar')
plt.legend(loc="center right")
plt.title('Variance schedule')
plt.xlabel('Diffusion steps')
plt.show()

snr = alphas_bar / one_minus_alphas_bar
plt.plot(np.log10(snr))
plt.title("SNR in log10 scale")
plt.xlabel('Diffusion steps')
plt.show()

diffusion_spaces = np.linspace(0, 1000, 11).astype(int)[:-1]
snr_log10_space = np.log10(snr)[diffusion_spaces][::-1]

image_old = Image.open(f"../datasets/data256x256/0000{1}.jpg")
img_tensor_old = img_to_tensor(image_old)
image_old_a_diff = [diffuse_t(img_tensor_old, t) for t in diffusion_spaces]
image_old_b_diff = [diffuse_t(img_tensor_old, t) for t in diffusion_spaces]

same_lpips = []
diff_lpips = []
num_samples = 100

for idx in tqdm(range(2, 2 + num_samples)):
    prefix = get_prefix(idx)

    # Read a PIL image
    image_new = Image.open(f"../datasets/data256x256/{prefix}{idx}.jpg")

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor_new = img_to_tensor(image_new)

    image_new_a_diff = [diffuse_t(img_tensor_new, t) for t in diffusion_spaces]
    image_new_b_diff = [diffuse_t(img_tensor_new, t) for t in diffusion_spaces]

    d_old = [loss_fn_vgg(image_old_a_diff[i], image_old_b_diff[i]).detach().numpy().squeeze() for i in
             range(len(diffusion_spaces))]
    d_new_old = [loss_fn_vgg(image_new_a_diff[i], image_old_a_diff[i]).detach().numpy().squeeze() for i in
                 range(len(diffusion_spaces))]

    image_old = image_new
    img_tensor_old = img_tensor_new
    image_old_a_diff = image_new_a_diff
    image_old_b_diff = image_new_b_diff

    same_lpips.append(d_old[::-1])
    diff_lpips.append(d_new_old[::-1])

same_lpips = np.mean(same_lpips, axis=0)
diff_lpips = np.mean(diff_lpips, axis=0)

plt.plot(snr_log10_space, same_lpips, label='LPIPS between same starting image')
plt.plot(snr_log10_space, diff_lpips, label='LPISP between different starting images')
plt.title('Average over 100 triplets')
plt.xlabel('SNR')
plt.xticks()
plt.ylabel('LPIPS distance')
plt.legend(loc='lower left')
plt.show()
