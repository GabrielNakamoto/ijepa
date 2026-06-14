from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from masks import apply_masks, generate_masks
import numpy as np
import matplotlib.pyplot as plt

def render_imgs_with_masks(imgs, mask_cfg):
  patch_size = mask_cfg['patch_size']
  pm, em = generate_masks((imgs.shape[0], 96//patch_size, 96//patch_size, 3), mask_cfg)
  fig, axes = plt.subplots(nrows=imgs.shape[0], ncols=6, figsize=(10,6))
  for i in range(imgs.shape[0]):
    render_img_with_masks(imgs[i], em[i], pm[i], axes[i], patch_size)
  fig.suptitle("Reconstructed patch masks for context and target encoders")
  plt.tight_layout()
  plt.show()

def render_img_with_masks(img, ctx_masks, target_masks, axs, patch_size):
  img = img.to(None).permute(2,1,0)
  axs[0].set_title("Original")
  axs[1].set_title("Context")
  axs[2].set_title("Targets")
  axs[0].imshow(img.numpy())
  axs[1].imshow(reconstruct_masked(img, ctx_masks, patch_size))
  for j in range(len(target_masks)):
    axs[2+j].imshow(reconstruct_masked(img, [target_masks[j]], patch_size))

def reconstruct_masked(img, mask, patch_size) -> np.ndarray:
  g = 96 // patch_size
  img = img.reshape(g, patch_size, g, patch_size, 3).permute(0, 2, 1, 3, 4).reshape(-1, patch_size*patch_size*3).unsqueeze(0)
  masked_img = apply_masks(img, mask)
  num_patches, feat = (96 // patch_size) ** 2, patch_size * patch_size * 3
  m = mask[0]
  idx = m.reshape(1, -1, 1).expand(1, m.shape[0], feat)
  reconstr = Tensor.zeros(1, num_patches, feat, dtype=dtypes.uchar) .scatter(1, idx, masked_img)
  reconstr = reconstr.reshape(g, g, patch_size, patch_size, 3).permute(0, 2, 1, 3, 4).reshape(96, 96, 3)
  return reconstr.numpy()
