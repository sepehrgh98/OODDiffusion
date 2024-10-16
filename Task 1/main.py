from SwinIR import SwinIR
import torch

# Load the SwinIR model architecture
model = SwinIR(
img_size= 64
, patch_size = 1
, in_chans = 3
, embed_dim = 180
, depths = [6, 6, 6, 6, 6, 6, 6, 6]
, num_heads = [6, 6, 6, 6, 6, 6, 6, 6]
, window_size = 8
, mlp_ratio = 2
, sf = 8
, img_range = 1.0
, upsampler = "nearest+conv"
, resi_connection = "1conv"
, unshuffle = True
, unshuffle_scale = 8
)

# Load checkpoint (.ckpt file)
checkpoint = torch.load('path_to_checkpoint.ckpt')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # Set model to evaluation mode