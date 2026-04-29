import math
import torch
from fast_hadamard_transform import fast_hadamard
# dim > 1
a = torch.randn(2,32,device="cuda",dtype=torch.float16)
scale = 1.0 / math.sqrt(a.size(-1))
print(a)
print(fast_hadamard(fast_hadamard(a,scale),scale))
# dim == 1
a = torch.randn(1024,device="cuda",dtype=torch.float16)
scale = 1.0 / math.sqrt(a.size(-1))
print(a)
print(fast_hadamard(fast_hadamard(a,scale),scale))
