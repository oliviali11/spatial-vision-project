# **Spatial Vision Project 2024 -- Jessica Huang**

This spatial vision project focuses on analyzing how well Monocular Depth Estimation Models like **MiDaS v.3.1** and **Depth-Anywhere** perform under different camera movements using images & videos we created under Blender and real life scenes. We run the models on our images and videos, and compare the model's normalized estimated depth vs. the real normalized depth value. 

You can take a look at some of the resulting graphs in the Jupyter Notebook located at depth_models/depth_mean.ipynb

### Monocular Depth Estimation Models
[Depth-Anywhere V2](https://github.com/LiheYoung/Depth-Anything)  
[MiDaS v3.1](https://github.com/isl-org/MiDaS)  

## Usage 
To visualize the depth of objects over time, upload the images, real positions of objects, annotation file, then we can do the following.  
  
**Run Depth-Anywhere**
```
python run.py --encoder <vits | vitb | vitl> --img-path <img-directory | single-img | txt-file> --outdir <outdir> [--pred-only] [--grayscale]
```
Arguments taken from [Depth-Anywhere github](https://github.com/LiheYoung/Depth-Anything):  
-    ```--img-path```  you can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.  
-    ```--pred-only```  is set to save the predicted depth map only. Without it, by default, we visualize both image and its depth map side by side.  
-    ```--grayscale```  is set to save the grayscale depth map. Without it, by default, we apply a color palette to the depth map.  
  
**Run MiDaS**
```
python run.py --model_type <model_type> --input_path input --output_path output
```
Arguments taken from [MiDaS github](https://github.com/isl-org/MiDaS):  
-    ```<model_type>``` is chosen from dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256, openvino_midas_v21_small_256.
-    I will be using dpt_beit_base_384


**Create graph on Jupyter Notebook**
Create a virtual environment, activate it. cd to directory with the notebook, open Jupyter Lab Notebook with ```jupyter lab```  
Once in the notebook, create graphs using the following command:  
```
dp.depth_process(<depth-model>, <object-movement>)
```
-    ```<model_type>``` is currently chosen from "depth_anywhere" or "midas"
-    ```<object-movement>``` - Example if I'm working with a car and panning over it, I would use ```"car_panover"```
