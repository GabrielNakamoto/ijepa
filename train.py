from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW
from tinygrad.nn import datasets
from tinygrad.nn.state import get_parameters
from tinygrad.engine.jit import TinyJit
import yaml

cfg = yaml.safe_load(open("config.yaml", "r"))
