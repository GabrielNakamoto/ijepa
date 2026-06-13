from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW
from tinygrad.nn import datasets
from tinygrad.nn.state import get_parameters
from tinygrad.engine.jit import TinyJit
from vit import ViT

batch_size, ishape, log_steps = 256, (32, 32, 3), 50
X_train, Y_train, X_test, Y_test = datasets.cifar()
X_train, X_test = X_train.permute(0,3,2,1), X_test.permute(0,3,2,1)
X_train, X_test = X_train.float() / 255.0, X_test.float() / 255.0
X_test, Y_test = X_test[:1000], Y_test[:1000] # limited by laptop VRAM

def random_flip(x:Tensor): return (Tensor.rand(x.shape[0],1,1,1) < 0.5).where(x.flip(1), x).contiguous()
def random_crop(x:Tensor,crop_size=32):
  B, w, h, c = x.shape
  # chw vs whc
  Xs = Tensor.randint(B, low=0, high=h-crop_size).reshape(B,1,1,1)
  Ys = Tensor.randint(B, low=0, high=w-crop_size).reshape(B,1,1,1)
  Xi = Tensor.arange(crop_size, dtype=dtypes.uint32).reshape(1,crop_size,1,1)
  Yi = Tensor.arange(crop_size, dtype=dtypes.uint32).reshape(1,1,crop_size,1)
  x_strip = (Xs + Xi).expand(-1, crop_size, h, c)
  y_strip = (Ys + Yi).expand(-1, crop_size, crop_size, c)
  return x.gather(1, x_strip).gather(2, y_strip)

# TODO: implement myself, batch-wise augmentation
# def cutmix(X, Y, alpha:float): pass

@TinyJit
def augment(X,Y):
  X = random_flip(X)
  # X = X.pad((None, None, (4,4),(4,4)))
  # X = random_crop(X)
  # todo: cutmix?
  return X, Y

def random_batch() -> tuple[Tensor, Tensor]:
  idx = Tensor.randint(batch_size, low=0, high=int(X_train.shape[0]))
  return augment(X_train[idx], Y_train[idx])

def one_cycle_lr(optim,total_steps,initial_lr,warmup_weight:float=0.3,max_lr:float=1e-3):
  warmup_steps = int(total_steps * warmup_weight)
  def fn(step):
    import math
    if step < warmup_steps: lr = initial_lr + (step / warmup_steps) * (max_lr - initial_lr)
    else: lr = initial_lr + 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))) * (max_lr - initial_lr)
    optim.lr.assign(Tensor([lr]))
  return fn

model = ViT(192, 3, 10, 10, 4, *X_train.shape[1:])
optim = AdamW(get_parameters(model))
total_steps = 1000
lr_scheduler = one_cycle_lr(optim,total_steps, 1e-5)

@TinyJit
def step(x, y):
  optim.zero_grad()
  logits = model(x)
  loss = logits.sparse_categorical_crossentropy(y).backward()
  optim.step()
  return loss

for n in range(total_steps):
  lr_scheduler(n)
  Tensor.training = True
  loss = step(*random_batch())
  Tensor.training = False
  if n % log_steps == 0:
    probs = model(X_test).softmax(axis=-1)
    acc = (Y_test == probs.argmax(-1)).mean()
    print(f"step={n}\t\tloss={loss.item():.2f}, valid acc={100*acc.item():.2f}%f")
  else:
    print(f"step={n}\t\tloss={loss.item():.2f}")
