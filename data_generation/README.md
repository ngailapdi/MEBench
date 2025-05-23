
## Download Blender 3.6 LTS

```
https://www.blender.org/download/releases/3-6/
```
Adjust Toys4k data path in `rendering/render_utils.py`

After adjusting the paths in `rendering/render_toys.sh` run the following command inside `rendering` to make sure everything is setup and things are working properly
```
bash render_toys.sh
```

# Codebase details
## 1. Scene config generation
The code for this is located in `scene_config_generation`. We want to generate a `.json` file for each scene, picking the objects, their pose, the lighting environment and floor, and the camera pose all ahead of time before rendering. A sample script to do this is `TOYS_create_scene_configs.py`.
```
python TOYS_create_scene_configs.py
```
## 2. Rendering
Run `render_toys.sh` inside `rendering` with the appropriate paths. 
```
bash render_toys.sh
```
# Generated data
Generated data can be found on [Hugging Face]{https://huggingface.co/ngailapdi}
