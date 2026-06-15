from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.state import tar_extract, get_parameters
from tinygrad.nn.optim import AdamW
from tinygrad.engine.jit import TinyJit
from ijepa import iJEPA
from masks import generate_masks
import yaml

cfg = yaml.safe_load(open("config.yaml", "r"))
enc_cfg, pred_cfg, mask_cfg = cfg['model']['encoder'], cfg['model']['predictor'], cfg['mask']
patch_size = cfg['mask']['patch_size']
num_patches = (96 // patch_size) ** 2
batch_size = 512

def load_stl10() -> Tensor: 
  tt = tar_extract(Tensor.from_url("http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz", gunzip=True))
  return tt['stl10_binary/unlabeled_X.bin'].reshape(-1, 3, 96, 96)#.to(None)

def random_batch_unsupervised():
  idx = Tensor.randint(batch_size, high=int(X.shape[0]))
  return X[idx], *generate_masks((batch_size, 96 // patch_size, 96 // patch_size, 3), mask_cfg)

X = load_stl10()
model = iJEPA((96,96,3), enc_cfg['embed_dim'], pred_cfg['embed_dim'], enc_cfg['depth'], pred_cfg['depth'], 12, cfg['mask']['patch_size'])
optim = AdamW(get_parameters(model))

@TinyJit
def step(imgs, enc_masks, pred_masks):
  def _smooth_l1_loss(x:Tensor, y:Tensor, beta:float=1.0):
    return ((x - y).abs() < beta).where(
      0.5 * (x - y) ** 2 / beta,
      (x - y).abs() - 0.5 * beta
    ).mean()

  optim.zero_grad()
  h, z = model(imgs, enc_masks, pred_masks)
  loss = _smooth_l1_loss(h, z).backward()
  optim.step()
  return loss

# 100k / batch_size = steps_per_epoch
target_epochs = 4
for i in range(target_epochs * (X.shape[0] // batch_size)):
  loss = step(*random_batch_unsupervised())
  print(f"step={i}, train loss={loss.item():.2f}")
