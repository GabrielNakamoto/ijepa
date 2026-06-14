from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, RMSNorm, Conv2d, LayerNorm
import math

def sincos_posemb_2d(w,h,d) -> Tensor: #(W*H,D)
  emb_h = sincos_posemb_1d(h,d)
  emb_w = sincos_posemb_1d(w,d)
  return emb_h.cat(emb_w)

def sincos_posemb_1d(n,d) -> Tensor: #(N,D)
  w_k = (1./1000) ** (2. / d * Tensor.arange(d//2))
  p_t = Tensor.arange(n)
  out = Tensor.einsum('m,d->md', p_t, w_k).reshape(n,d//2)
  return out.sin().stack(out.cos(), dim=-1).reshape(n,d)

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
  def __init__(self, patch_size:int, dim:int, n_heads:int, depth:int, img_shape):
    h, w, c = img_shape
    self.dim, self.N = dim, int((h*w)/(patch_size*patch_size))
    self.blocks = [TransformerBlock(dim, n_heads) for _ in range(depth)]
    self.patch_emb = Conv2d(c, dim, kernel_size=patch_size, stride=patch_size)
    pn = int(math.sqrt(self.N))
    self.pos_emb = sincos_posemb_2d(pn, pn, dim)
    self.out_norm = RMSNorm(dim)
  def __call__(self, img:Tensor, masks=None, dropout_p:float=0.1) -> Tensor:
    # img (B,H,W,C)
    B = img.shape[0]
    x = self.patch_emb(img).flatten(2).transpose(1,2) # (B,N,D)
    x = x + self.pos_emb

    if masks is not None: # apply masks
      pass

    for blk in self.blocks: x = blk(x, dropout_p=dropout_p)
    return self.out_norm(x)

class ViTPredictor:
  def __init__(self, num_patches:int, ctx_dim:int, pred_dim:int, n_heads:int, depth:int):
    self.pred_emb = Linear(ctx_dim, pred_dim)
    self.blocks = [TransformerBlock(pred_dim, n_heads) for _ in range(depth)]
    self.out_norm = RMSNorm(pred_dim)
  def __call__(self, x:Tensor, masks_x, masks, dropout_p:float=0.1) -> Tensor:
    x = self.pred_emb(x)

    for blk in self.blocks: x = blk(x, dropout_p=dropout_p)
    x = self.out_norm(x)
    return x

# https://arxiv.org/pdf/2301.08243
class iJEPA:
  def __init__(self, img_shape, enc_dim:int, pred_dim:int, enc_depth:int, pred_depth:int, n_heads:int, num_patches:int, patch_size:int):
    self.encoder, self.target_encoder = ViT(patch_size, enc_dim, n_heads, enc_depth, img_shape), ViT(patch_size, enc_dim, n_heads, enc_depth, img_shape)
    self.predictor = ViTPredictor(num_patches, enc_dim, pred_dim, n_heads, pred_depth)
  def __call__(self, imgs:Tensor, masks_enc, masks_pred):
    target = [ self.target_encoder, LayerNorm ]
    h = imgs.sequential(target).detach()
    z = self.encoder(imgs, masks_enc)
