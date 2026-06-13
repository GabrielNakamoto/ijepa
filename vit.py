from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, RMSNorm, Embedding

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
  def __init__(self, dim, n_heads, layers, num_classes, P, w, h, c):
    self.dim, self.num_class, self.N = dim, num_classes, int((h*w)/(P*P))
    self.blocks = [TransformerBlock(dim, n_heads) for _ in range(layers)]
    self.patch_emb = Linear(P*P*c, dim)
    self.pos_emb = Embedding(self.N + 1, dim)
    self.out_norm = RMSNorm(dim)
    self.x_class = Tensor.zeros(1,1, dim)
    self.head = Linear(dim, num_classes)
  def __call__(self, img:Tensor, dropout_p:float=0.1) -> Tensor:
    B = img.shape[0]
    img = img.reshape(B, self.N, -1)
    _cls = self.x_class.expand(B, 1, self.dim)
    x = _cls.cat(self.patch_emb(img), dim=1) + self.pos_emb(Tensor.arange(self.N + 1))
    for t in self.blocks: x = t(x, dropout_p=dropout_p)
    x = self.out_norm(x)
    return self.head(x[:, 0])
