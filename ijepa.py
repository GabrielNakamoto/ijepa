from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, LayerNorm, RMSNorm, Conv2d
import math
from masks import apply_masks

def sincos_posemb_2d(w,h,d):
  grid_x, grid_y = Tensor.arange(w).meshgrid(Tensor.arange(h))
  emb_h = sincos_posemb_1d_from_grid(grid_x,d//2)
  emb_w = sincos_posemb_1d_from_grid(grid_y,d//2)
  return emb_h.cat(emb_w, dim=-1)

def sincos_posemb_1d_from_grid(grid,d) -> Tensor: #(N,D)
  w_k = (1./1000) ** (2. / d * Tensor.arange(d//2))
  out = Tensor.einsum('m,d->md', grid.flatten(), w_k).reshape(-1,d//2) # outer product
  return out.sin().cat(out.cos(), dim=-1)

class TransformerBlock:
  def __init__(self, dim, n_heads):
    self.dim = dim
    self.n_heads = n_heads
    self.head_dim = dim // n_heads
    self.qkv_proj = Linear(dim, dim*3)
    self.attn_norm = RMSNorm(dim)
    self.mlp = [Linear(dim, dim * 4), Tensor.swish, Linear(dim * 4, dim)]
    self.mlp_norm = RMSNorm(dim)
    self.attn_proj = Linear(dim, dim)
  def _attention(self, x:Tensor, dropout_p:float=0.0) -> Tensor:
    B, N = x.shape[:2]
    q, k, v = [t.squeeze(2).transpose(1,2) for t in self.qkv_proj(x).reshape(B,N,3,self.n_heads,self.head_dim).split(1, dim=2)]
    attn = q.scaled_dot_product_attention(k, v, dropout_p=dropout_p)
    return self.attn_proj(attn.reshape((B,N,self.dim)))
  def __call__(self, x:Tensor, dropout_p:float=0.0) -> Tensor:
      x = x + self._attention(self.attn_norm(x), dropout_p=dropout_p)
      return x + self.mlp_norm(x).sequential(self.mlp).dropout(dropout_p)

# https://arxiv.org/pdf/2010.11929
class ViT:
  def __init__(self, num_patches:int, patch_size:int, dim:int, n_heads:int, depth:int, img_shape):
    _, _, c = img_shape
    self.dim = dim
    self.blocks = [TransformerBlock(dim, n_heads) for _ in range(depth)]
    self.patch_emb = Conv2d(c, dim, kernel_size=patch_size, stride=patch_size)
    pn = int(math.sqrt(num_patches))
    self.pos_emb = sincos_posemb_2d(pn, pn, dim)
    self.out_norm = LayerNorm(dim)
  def __call__(self, img:Tensor, masks:list[Tensor]|None=None, dropout_p:float=0.1) -> Tensor:
    # img (B,C,W,H)
    x = self.patch_emb(img).flatten(2).transpose(1,2) # (B,N,D)
    x = x + self.pos_emb
    if masks is not None: x = apply_masks(x, masks) # (B, masked patches, D)
    for blk in self.blocks: x = blk(x, dropout_p=dropout_p)
    return self.out_norm(x)

class ViTPredictor:
  def __init__(self, num_patches:int, ctx_dim:int, pred_dim:int, n_heads:int, depth:int):
    self.dim = pred_dim
    self.pred_emb = Linear(ctx_dim, pred_dim)
    self.blocks = [TransformerBlock(pred_dim, n_heads) for _ in range(depth)]
    self.out_norm = LayerNorm(pred_dim)
    pn = int(math.sqrt(num_patches))
    self.pos_emb = sincos_posemb_2d(pn,pn,pred_dim)
    self.num_patches = num_patches
    self.mask_token = Tensor.zeros(1, 1, pred_dim)
  def __call__(self, x:Tensor, ctx_masks, pred_masks, dropout_p:float=0.1) -> Tensor:
    B = x.shape[0] // len(ctx_masks) # batch size?
    x = self.pred_emb(x)
    p = self.pos_emb.repeat(B, 1, 1)
    x += apply_masks(p, ctx_masks) # (batch size, n masked patches, pred dim)

    pos_emb = self.pos_emb.repeat(B, 1, 1) # (batch size, )
    pos_emb = apply_masks(pos_emb, pred_masks)
    pos_emb = pos_emb.repeat_interleave(len(ctx_masks), dim=0) # ()
    pred_tokens = self.mask_token.repeat(pos_emb.shape[0], pos_emb.shape[1], 1)
    pred_tokens += pos_emb
    x = x.repeat(len(pred_masks), 1, 1)
    x = x.cat(pos_emb, dim=1)

    for blk in self.blocks: x = blk(x, dropout_p=dropout_p)
    x = self.out_norm(x)
    return x

# https://arxiv.org/pdf/2301.08243
class iJEPA:
  def __init__(self, img_shape, enc_dim:int, pred_dim:int, enc_depth:int, pred_depth:int, n_heads:int, patch_size:int):
    h, w, _ = img_shape
    num_patches = (h // patch_size) * (w // patch_size)
    self.encoder = ViT(num_patches, patch_size, enc_dim, n_heads, enc_depth, img_shape)
    self.target_encoder = ViT(num_patches, patch_size, enc_dim, n_heads, enc_depth, img_shape)
    self.predictor = ViTPredictor(num_patches, enc_dim, pred_dim, n_heads, pred_depth)
  def __call__(self, imgs:Tensor, masks_enc:list[Tensor], masks_pred:list[Tensor]) -> tuple[Tensor, Tensor]:
    # compute target representation
    h = self.target_encoder(imgs).detach().layernorm(-1) # normalize feature dim
    h = apply_masks(h, masks_enc).repeat_interleave(len(masks_enc))
    # predict representation from context
    z = self.encoder(imgs, masks=masks_enc)
    z = self.predictor(z, masks_enc, masks_pred)
    return h, z
