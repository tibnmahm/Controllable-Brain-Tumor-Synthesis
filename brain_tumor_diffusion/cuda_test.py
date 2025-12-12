import torch
print(torch.__version__)
print(torch.version.cuda)        # Should show 12.1
print(torch.cuda.is_available()) # Should be True