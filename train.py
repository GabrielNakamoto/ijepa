from tinygrad.tensor import Tensor
from tinygrad.nn.state import tar_extract
import matplotlib.pyplot as plt
from ijepa import iJEPA
import yaml

from masks import generate_masks

cfg = yaml.safe_load(open("config.yaml", "r"))

# returns a batch of 10k unlabelled images in (H,W,C) form
def load_stl10(split:int) -> Tensor: 
  tt = tar_extract(Tensor.from_url("http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz", gunzip=True))
  return tt['stl10_binary/unlabeled_X.bin'].reshape(-1, 3, 96, 96).chunk(10)[split].squeeze(0).permute(0,3,2,1).to(None)

model = iJEPA((92,92,3), cfg['model'])

X = load_stl10(1)
