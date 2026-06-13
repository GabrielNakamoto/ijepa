from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW
from tinygrad.nn import datasets
from tinygrad.nn.state import get_parameters
from tinygrad.engine.jit import TinyJit
from vit import ViT

batch_size, ishape, log_steps = 256, (32, 32, 3), 50
X_train, Y_train, X_test, Y_test = datasets.cifar()
Y_train = Y_train.one_hot(10)
X_train, X_test = X_train.permute(0,2,3,1), X_test.permute(0,2,3,1)
X_train, X_test = X_train.float() / 255.0, X_test.float() / 255.0
X_test, Y_test = X_test[:1000], Y_test[:1000] # limited by laptop VRAM

def random_flip(x:Tensor): return (Tensor.rand(x.shape[0],1,1,1) < 0.5).where(x.flip(2), x).contiguous()
def random_crop(x:Tensor,crop_size=32):
  B, w, h, c = x.shape
  Xs = Tensor.randint(B, low=0, high=h-crop_size).reshape(B,1,1,1)
  Ys = Tensor.randint(B, low=0, high=w-crop_size).reshape(B,1,1,1)
  Xi = Tensor.arange(crop_size, dtype=dtypes.uint32).reshape(1,1,crop_size,1)
  Yi = Tensor.arange(crop_size, dtype=dtypes.uint32).reshape(1,crop_size,1,1)
  x_strip = (Xs + Xi).expand(-1, h, crop_size, c)
  y_strip = (Ys + Yi).expand(-1, crop_size, crop_size, c)
  return x.gather(2, x_strip).gather(1, y_strip)

def make_square_mask(shape, mask_size) -> Tensor:
  BS, H, W, _ = shape
  low_x = Tensor.randint(BS, low=0, high=W-mask_size).reshape(BS,1,1,1)
  low_y = Tensor.randint(BS, low=0, high=H-mask_size).reshape(BS,1,1,1)
  idx_x = Tensor.arange(W, dtype=dtypes.int32).reshape((1,1,W,1))
  idx_y = Tensor.arange(H, dtype=dtypes.int32).reshape((1,H,1,1))
  return (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))

def cutmix(X, Y, perms, mask_size):
  mask = make_square_mask(X.shape, mask_size)
  lamb = float(mask_size**2)/(X.shape[1]*X.shape[2])
  return mask.where(X[perms], X), lamb * Y[perms] + (1. - lamb) * Y

@TinyJit
def augment(X,Y, mask_size=16):
  X = random_flip(X)
  X = X.pad((None, (4,4),(4,4), None))
  X = random_crop(X)
  perms = Tensor.randperm(int(X.shape[0]), device=X.device)
  X, Y = X[perms], Y[perms]
  return cutmix(X, Y, perms, mask_size)

def random_batch() -> tuple[Tensor, Tensor]:
  idx = Tensor.randint(batch_size, low=0, high=int(X_train.shape[0]))
  return augment(X_train[idx], Y_train[idx])

# https://docs.pytorch.org/docs/2.12/generated/torch.optim.lr_scheduler.OneCycleLR.html
def one_cycle_lr(optim,total_steps,initial_lr,warmup_weight:float=0.3,max_lr:float=1e-3):
  warmup_steps = int(total_steps * warmup_weight)
  def fn(step):
    import math
    if step < warmup_steps: lr = initial_lr + (step / warmup_steps) * (max_lr - initial_lr)
    else: lr = initial_lr + 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))) * (max_lr - initial_lr)
    optim.lr.assign(Tensor([lr]))
  return fn

model = ViT(192, 3, 10, 10, 4, *X_train.shape[1:])
optim = AdamW(get_parameters(model), weight_decay=0.05)
total_steps = 1000
lr_scheduler = one_cycle_lr(optim,total_steps, 5e-4)

@TinyJit
def step(x, y):
  optim.zero_grad()
  logits = model(x)
  loss = logits.cross_entropy(y).backward()
  optim.step()
  return loss

for n in range(total_steps):
  lr_scheduler(n)
  Tensor.training = True
  loss = step(*random_batch())
  Tensor.training = False
  if n % log_steps == 0:
    logits = model(X_test)
    valid_loss = logits.sparse_categorical_crossentropy(Y_test)
    acc = (Y_test == logits.softmax(axis=-1).argmax(-1)).mean()
    print(f"step={n}\t\tloss={loss.item():.2f}, valid_loss={valid_loss.item():.2f}, valid acc={100*acc.item():.2f}%f")
