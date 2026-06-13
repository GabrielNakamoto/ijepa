from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import math

def sample_block_size(x, scale, aspect_ratio_scale):
  _rand = Tensor.rand(1, device=x.device).item()
  _, H, W, _ = x.shape
  min_s, max_s = scale
  min_ar, max_ar = aspect_ratio_scale
  mask_scale = min_s + _rand * (max_s - min_s)
  max_keep = int(H * W * mask_scale)
  aspect_ratio = min_ar + _rand * (max_ar - min_ar)
  h = int(round(math.sqrt(max_keep * aspect_ratio)))
  w = int(round(math.sqrt(max_keep / aspect_ratio)))
  return min(H - 1, h), min(W - 1, w)

def sample_block_mask(x, b_size, min_keep=4, acceptable_regions=None):
  h, w = b_size
  _, H, W, _ = x.shape
  while True:
    top = Tensor.randint(1, low=0, high=H-h)
    left = Tensor.randint(1, low=0, high=W-w)
    mask = Tensor.zeros(H,W,dtype=dtypes.uint32)
    mask[top:top+h, left:left+w] = 1

    if acceptable_regions is not None:
      for r in acceptable_regions: mask *= r

    mask = mask.flatten().nonzero()
    if mask.numel() >= min_keep:
      mask = mask.squeeze()
      mask_c = Tensor.ones(H,W,dtype=dtypes.uint32)
      mask_c[top:top+h, left:left+w] = 0
      return mask, mask_c

def generate_masks(X:Tensor, pred_scales, enc_scales, npreds:int, nenc:int) -> tuple[Tensor, Tensor]:
  def _stack(T:list[Tensor]): return T[0].stack(*T[1:])
  plen = elen = X.shape[1]*X.shape[2]
  pred_masks, enc_masks = [], []
  for _ in range(X.shape[0]):
    p_size = sample_block_size(X, *pred_scales)
    e_size = sample_block_size(X, *enc_scales)
    Mp, Mc, Me = [], [], []
    # choose prediction masks first so context doesn't overlap
    for _ in range(npreds):
      mp, mc = sample_block_mask(X, p_size)
      plen = min(plen, mp.numel())
      Mp.append(mp); Mc.append(mc)
    for _ in range(nenc):
      me, _ = sample_block_mask(X, e_size, acceptable_regions=Mc)
      elen = min(elen, me.numel())
      Me.append(me)
    Mp, Me = _stack(Mp), _stack(Me)
    Mp, Me = Mp[:,:plen], Me[:,:elen]
    pred_masks.append(Mp); enc_masks.append(Me)
  return _stack(pred_masks), _stack(enc_masks)

def apply_masks(X:Tensor, masks):
  pass
