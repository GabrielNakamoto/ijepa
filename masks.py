from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import math, random
import numpy as np

def sample_block_size(shape, scale: tuple[int, int], aspect_ratio_scale) -> tuple[int, int]:
  _rand = random.random()
  _, H, W, _ = shape
  min_s, max_s = scale
  min_ar, max_ar = aspect_ratio_scale
  mask_scale = min_s + _rand * (max_s - min_s)
  max_keep = int(H * W * mask_scale)
  aspect_ratio = min_ar + _rand * (max_ar - min_ar)
  h = int(round(math.sqrt(max_keep * aspect_ratio)))
  w = int(round(math.sqrt(max_keep / aspect_ratio)))
  return min(H - 1, h), min(W - 1, w)

# constructs binary mask tensor, returns with conjugate
def sample_block_mask(shape, b_size:tuple[int,int], min_keep=4, acceptable_regions=None) -> tuple[np.ndarray, np.ndarray]:
  h, w = b_size
  _, H, W, _ = shape
  while True:
    top = random.randint(0, int(H)-h)
    left = random.randint(0, int(W)-w)
    mask = np.zeros((H,W),np.uint32)
    mask[top:top+h, left:left+w] = 1

    if acceptable_regions is not None:
      for r in acceptable_regions: mask *= r

    mask = np.flatnonzero(mask)
    if mask.size >= min_keep:
      mask_c = np.ones((H,W),np.uint32)
      mask_c[top:top+h, left:left+w] = 0
      return mask.squeeze(), mask_c

# shape (batch, patch height grid, patch width grid, channels)
def generate_masks(shape, cfg:dict) -> tuple[list[Tensor], list[Tensor]]:
  plen = elen = shape[1]*shape[2]
  pred_masks, enc_masks = [[]] * cfg['num_pred_masks'], [[]] * cfg['num_enc_masks']
  for _ in range(shape[0]):
    # print(f"Generated mask {i+1}/{shape[0]}", end='\r', flush=True)
    p_size = sample_block_size(shape, cfg['pred_mask_scale'], cfg['aspect_ratio'])
    e_size = sample_block_size(shape, cfg['enc_mask_scale'], (1.,1.))
    Mc = []
    # choose prediction masks first so context doesn't overlap
    for i in range(cfg['num_pred_masks']):
      mp, mc = sample_block_mask(shape, p_size)
      plen = min(plen, mp.size)
      pred_masks[i].append(mp)
      Mc.append(mc)

    for i in range(cfg['num_enc_masks']):
      me, _ = sample_block_mask(shape, e_size, acceptable_regions=Mc)
      elen = min(elen, me.size)
      enc_masks[i].append(me)

  pred_masks = [Tensor(np.stack([p[:plen] for p in pp], axis=0)) for pp in pred_masks]
  enc_masks = [Tensor(np.stack([e[:elen] for e in ee], axis=0)) for ee in enc_masks]
  return pred_masks, enc_masks

def apply_masks(X:Tensor, masks:list[Tensor]) -> Tensor:
  # X(bchsz, num_patches, feature dim)
  def _proc(m): return X.gather(1, m.unsqueeze(-1).repeat(1,1,X.shape[-1]))
  xs = [_proc(m) for m in masks]
  return xs[0] if len(xs) == 1 else xs[0].cat(*xs[1:], dim=0)
