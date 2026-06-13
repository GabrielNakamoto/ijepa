from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import math

def sample_block_size(x:Tensor, scale: tuple[int, int], aspect_ratio_scale) -> tuple[int, int]:
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

# constructs binary mask tensor, returns with conjugate
def sample_block_mask(x:Tensor, b_size:tuple[int,int], min_keep=4, acceptable_regions=None) -> tuple[Tensor, Tensor]:
  h, w = b_size
  _, H, W, _ = x.shape
  while True:
    top = Tensor.randint(1, low=0, high=int(H)-h)
    left = Tensor.randint(1, low=0, high=int(W)-w)
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

def generate_masks(X:Tensor, cfg:dict) -> tuple[Tensor, Tensor]:
  def _stack(T:list[Tensor]): return T[0].stack(*T[1:])
  plen = elen = X.shape[1]*X.shape[2]
  pred_masks, enc_masks = [], []
  for _ in range(X.shape[0]):
    p_size = sample_block_size(X, cfg['pred_mask_scale'], cfg['aspect_ratio'])
    e_size = sample_block_size(X, cfg['enc_mask_scale'], (1.,1.))
    Mp, Mc, Me = [], [], []
    # choose prediction masks first so context doesn't overlap
    for _ in range(cfg['num_pred_masks']):
      mp, mc = sample_block_mask(X, p_size)
      plen = min(plen, mp.numel())
      Mp.append(mp); Mc.append(mc)

    for _ in range(cfg['num_enc_masks']):
      me, _ = sample_block_mask(X, e_size, acceptable_regions=Mc)
      elen = min(elen, me.numel())
      Me.append(me)

    Mp, Me = _stack(Mp), _stack(Me)
    Mp, Me = Mp[:,:plen], Me[:,:elen]
    pred_masks.append(Mp); enc_masks.append(Me)
  return _stack(pred_masks), _stack(enc_masks)

def apply_masks(X:Tensor, masks:Tensor):
  # X(bchsz, num_patches, dim)
  # masks(num masks, num_patches)
  pass
