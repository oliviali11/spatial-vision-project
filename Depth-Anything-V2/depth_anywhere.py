# import cv2
# import torch
# import numpy as np
# from depth_anything_v2.dpt import DepthAnythingV2

# # Initialize and load the DepthAnythingV2 model
# model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
# model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitb.pth', map_location='cpu'))
# model.eval()

# # Read the input image
# raw_img = cv2.imread('corrupted_images/snow/1/0001_snow_s1.png')

# # Perform depth estimation inference
# depth = model.infer_image(raw_img)  # Returns a HxW raw depth map (NumPy array or tensor)

# # Normalize the depth map for visualization purposes (optional)
# depth_normalized = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX)

# # Convert the normalized depth map to an 8-bit image for saving
# depth_normalized = depth_normalized.astype(np.uint8)

# # Save the depth map as a PNG image
# cv2.imwrite("predicted_depth_maps/depth_estimation_result.png", depth_normalized)

# # Optionally, save the depth map as a .npy file for further analysis
# np.save("predicted_depth_maps/depth_estimation_result.npy", depth)

# import cv2
# import torch
# import numpy as np
# from depth_anything_v2.dpt import DepthAnythingV2

# # Initialize and load the DepthAnythingV2 model on the CPU
# model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
# model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitb.pth', map_location='cpu'))
# model.eval()

# # Read the input image
# raw_img = cv2.imread('corrupted_images/snow/1/0001_snow_s1.png')

# # Convert the image to a tensor, permuting dimensions from HxWxC to CxHxW and adding a batch dimension
# raw_img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0).float()

# # Perform depth estimation inference on the CPU
# with torch.no_grad():
#     depth = model.infer_image(raw_img_tensor)  # Returns a HxW raw depth map

# # Convert the depth map tensor to a NumPy array
# depth = depth.cpu().squeeze().numpy()

# # Normalize the depth map for visualization purposes (optional)
# depth_normalized = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX)

# # Convert the normalized depth map to an 8-bit image for saving
# depth_normalized = depth_normalized.astype(np.uint8)

# # Save the depth map as a PNG image
# cv2.imwrite("predicted_depth_maps/depth_estimation_result.png", depth_normalized)

# # Optionally, save the depth map as a .npy file for further analysis
# np.save("predicted_depth_maps/depth_estimation_result.npy", depth)


# import cv2
# import torch

# from depth_anything_v2.dpt import DepthAnythingV2

# # Initialize and load the DepthAnythingV2 model
# model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
# model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitb.pth', map_location='cpu'))
# model.eval()

# # Read the input image
# raw_img = cv2.imread('corrupted_images/snow/1/0001_snow_s1.png')

# # Convert the image to RGB (OpenCV loads images in BGR format)
# raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

# # Convert the image to a NumPy array and then a PyTorch tensor, forcing it to the CPU
# raw_img_tensor = torch.from_numpy(raw_img_rgb).permute(2, 0, 1).unsqueeze(0).float().cpu()

# # Perform depth estimation inference
# depth = model.infer_image(raw_img_tensor)

import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb' # or 'vits', 'vitl', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('corrupted_images/snow/1/0001_snow_s1.png')
depth = model.infer_image(raw_img)

depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_normalized = depth_normalized.astype(np.uint8)

output_dir = 'predicted_depth_maps'

cv2.imwrite(f"{output_dir}/depth_estimation_result.png", depth_normalized)

np.save(f"{output_dir}/depth_estimation_result.npy", depth)










