'''
feakilename = "/home/sstojanov3/develop/CRIBpp_generic/rendering/generate.py"
exec(compile(open(filename).read(), filename, 'exec'))                                      
'''

import bpy
import numpy as np
import colorsys
import json
import os
import sys
import argparse
import time
import copy
from mathutils import Matrix, Vector

### ugly but necessary because of Blender's Python
fpath = bpy.data.filepath
# root_path = os.path.dirname(os.path.abspath(__file__))
# root_path = '/'.join(fpath.split('/')[:-2])
# blend_file_dir_path = os.path.join(root_path, "common")
# python_file_dir_path = os.path.join(root_path, 'rendering')

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script's directory
os.chdir(script_dir)
root_path = '/'.join(script_dir.split('/')[:-1])
blend_file_dir_path = os.path.join(root_path, "common")
python_file_dir_path = os.path.join(root_path, 'rendering')
# python_file_dir_path = script_dir

# Optional: Update Blender's relative paths
# bpy.context.preferences.filepaths.script_directory = script_dir

# Print the current working directory to confirm
print(f"Current working directory set to: {os.getcwd()}")
print('python path: ', python_file_dir_path)

sys.path.append(blend_file_dir_path)
sys.path.append(python_file_dir_path)

import render_utils

bpy.context.scene.display_settings.display_device = 'sRGB'
bpy.context.scene.view_settings.view_transform = 'Standard'
print("Color management reset to Standard.")
bpy.context.scene.view_layers["ViewLayer"].name = "View Layer"

print(bpy.context.scene.render.image_settings.file_format)

def main():
    ### load datagen params
    with open(os.path.join(python_file_dir_path, "data_generation_parameters.json")) as load_file:
        data_gen_params = json.load(load_file)
    
    cam_params = data_gen_params["camera_parameters"]
    render_params = data_gen_params["render_parameters"]
    

    bpy.data.scenes[0].render.engine = "CYCLES"
    # bpy.data.scenes[0].render.engine = "BLENDER_EEVEE"


    # Set the device_type
    # bpy.context.preferences.addons[
    #     "cycles"
    # ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # # Set the device and feature set
    # bpy.context.scene.cycles.device = "GPU"

    # # get_devices() to let Blender detects GPU device
    # bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    # for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    #     if d.type == "GPU":
    #         d["use"] = 1 # Using all devices, include GPU and CPU
    #     print(d["name"], d["use"])

    
    bpy.context.scene.render.engine = "CYCLES"

    # For now, denoiser is always turned on, but the  _used_
    # bpy.context.scene.cycles.use_denoising = denoise
    # if denoise:
    #     try:
    #         bpy.context.scene.cycles.denoiser = "OPTIX"
    #     except Exception as e:
    #         logger.warning(f"Cannot use OPTIX denoiser {e}")

    # bpy.context.scene.cycles.samples = num_samples  # i.e. infinity
    # bpy.context.scene.cycles.adaptive_min_samples = min_samples
    # bpy.context.scene.cycles.adaptive_threshold = (
    #     adaptive_threshold  # i.e. noise threshold
    # )
    # bpy.context.scene.cycles.time_limit = time_limit
    # bpy.context.scene.cycles.film_exposure = exposure
    bpy.context.scene.cycles.volume_step_rate = 0.1
    bpy.context.scene.cycles.volume_preview_step_rate = 0.1
    bpy.context.scene.cycles.volume_max_steps = 32
    bpy.context.scene.cycles.volume_bounces = 4
    
    ### read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_path", type=str)
    
    #argv = "--dataset_type toys --output_path /home/sstojanov3/develop/CRIBpp_generic/toys_rendering_output --config_path /home/sstojanov3/develop/CRIBpp_generic/common/toys_scene_configs/00029.json --dataset_path /data/TOYS4K_BLEND_FILES_PACKED_V1"
    #argv = argv.split(' ')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    # adding floor plane, material to it and background
    # render_utils.add_plane((0,0,0), 10.0)
    
    # pbr_name = render_utils.add_PBR(
    #     'floor_material', 
    #     config['appearance_assets']['floor_pbr'],
    #     os.path.join(blend_file_dir_path, "assets", "pbr")
    #     )
    # bpy.data.objects['floor_object'].data.materials.append(bpy.data.materials['floor_material'])
    
    # hdr_name, map_node, dome = render_utils.add_IBL(
    #         config['appearance_assets']['background_hdr'],
    #         os.path.join(blend_file_dir_path, "assets", "HDRI_haven_raw"),
    #         config['environment_strength']
    #         )
   
    ## loading objects
    scn = bpy.context.scene

    load_end_time = load_start_time = time.time()

    ################### HARDCODE
    # config["objects"] = []
    ################### HARDCODE
    key = [k for k in bpy.data.objects.keys() if 'floor' in k][0]
    floor = bpy.data.objects[key]
    floor_location  = floor.location
    shiftz = floor_location.z
    shiftx = floor_location.x
    shifty = floor_location.y 

    existing_obj_name_list = [obj.name for obj in bpy.data.objects]
    existing_obj_list = [bpy.data.objects[name] for name in existing_obj_name_list]

    
    apply_customed_mat = False
    for obj_dict in config["objects"]:
    
        # nothing is active, nothing is selected
        bpy.context.view_layer.objects.active = None

        for o in bpy.data.objects:
            o.select_set(False)
        
        load_start_time = time.time()
        
        obj_path = os.path.join(args.dataset_path, obj_dict["obj_subpath"])
        obj = render_utils.load_obj(scn, obj_path, args.dataset_type)
        
        load_end_time = time.time()
        # make loaded object active and selected
        obj.select_set(True)  
        bpy.context.view_layer.objects.active = obj
       
        # clear normals
        bpy.ops.mesh.customdata_custom_splitnormals_clear()

        # recompute normals
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.editmode_toggle()
        
        ### rescaling object to fit in unit cube
        vertices = np.array([v.co for v in obj.data.vertices])
        if len(vertices) == 0 or 'Arch_Cube' in obj_dict['obj_subpath'] or 'Arch Cube' in obj_dict['obj_subpath']:
            vertices, obj = render_utils.get_geometry_nodes_vertices(obj)
            apply_customed_mat = True
        obj.scale = obj.scale * 0.5 / np.max(np.abs(vertices))
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)
        
        ### rescaling object to actual desired scale

        obj.scale = [obj_dict["obj_scale"]]*3
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        ### calculate z_shift to put object on plane surface
        pose_mat = np.eye(4,4) # empty variable where we will store the pose matrix
        rot_mat = np.array(obj_dict["obj_pose"]) #3x3 pose rotation matrix
        pose_mat[:3,:3] = rot_mat #set the rotation part of the pose matrix
        
        # compute how much to shift the object up so it appears on top of the surface
        if apply_customed_mat:
            z_shift = shiftz
        else:
            vertices = np.array([v.co for v in obj.data.vertices])
            vertices_w_co = (rot_mat @ vertices.T).T
            z_shift = np.abs(vertices_w_co[:,2].min()) + shiftz

        

        ### set location in the pose matrix
        pose_mat[0,-1] = obj_dict['obj_location'][0] + shiftx
        pose_mat[1,-1] = obj_dict['obj_location'][1] + shifty
        pose_mat[2,-1] = z_shift
        
        ### set pose matrix
        obj.matrix_world = Matrix(pose_mat)
        #obj.location = (obj_dict['obj_location'][0],
        #                obj_dict['obj_location'][1],
        #                z_shift)

        if args.dataset_type == 'ABC' or apply_customed_mat:

            mat_name = render_utils.add_configured_material(bpy.data, obj_dict['obj_color'])
            render_utils.assign_material(obj, mat_name)
    
    
    bpy.context.view_layer.update()
    ## rendering settings
    render_utils.apply_settings(render_params, scn)
    
    # where to output renders
    #####################################################
    obj_name_list = [obj.name for obj in bpy.data.objects if obj.name not in existing_obj_name_list]
    obj_list = [bpy.data.objects[name] for name in obj_name_list]
    print('Checking obj collisions')
    updated_existing_obj_list = render_utils.check_and_remove_colliding_objects(existing_obj_list, obj_list)
    output_dirpath = os.path.join(args.output_path, args.config_path.split('/')[-1].replace('.json',''))
    # try:



    # Add camera
    cam = render_utils.add_camera(cam_params)
    scn.camera = cam
    constraint_object, parent_object = render_utils.constrain_camera(cam, location=(shiftx,shifty,shiftz))
    scn.frame_start = 0

    # set camera position
    cam_positions = np.array(config['camera']['positions'])
    print(cam_positions)
    cam_positions[:,0] += shiftx
    cam_positions[:,1] += shifty

    print('Checking camera collisions')


    updated_existing_obj_list = render_utils.check_and_remove_camera_colliding_objects(cam_positions, updated_existing_obj_list)
    obj_list = obj_name_list
    
    n_frames = len(cam_positions)
    scn.frame_end = n_frames
    cam_rotations = []

    render_utils.do_compositing(output_dirpath, obj_name_list)
    deg = np.random.choice([0,30,60,90,120,150,180,210,240,270,300,330])
    # try:
    #     dome.location[0] = np.random.uniform(-2.5,2.5)
    #     dome.location[1] = np.random.uniform(-2.5,2.5)
    #     dome.rotation_euler = (0, 0, -np.radians(deg))
    # except:
    #     pass
    for i in np.arange(n_frames):
        scn.frame_set(i)
        parent_object.location = cam_positions[i]
        parent_object.keyframe_insert(data_path='location', frame = i)
        
        #cam.rotation_euler = (0,0,np.radians(np.random.uniform(-30,30)))
        #cam.keyframe_insert(data_path='rotation_euler', frame=i)

        # constraint_object.location = config['camera']['track_to_point'][i]
        # constraint_object.location = config['camera']['track_to_point'][-1]

        #constraint_object.location = (
                #np.array(obj.location) + np.random.uniform(-0.3,0.3,3)
            #)
        
        # rotating the environment
        # deg = np.random.choice([0,30,60,90,120,150,180,210,240,270,300,330])

        # map_node.inputs['Rotation'].default_value = (0,0,np.radians(deg))
        # map_node.inputs['Rotation'].keyframe_insert(data_path="default_value",frame=i)



        # try:
        #     dome.keyframe_insert(data_path='location', frame=i)
        #     dome.keyframe_insert(data_path='rotation_euler', frame=i)
        # except:
        #     pass



        # constraint_object.location = obj.location
        # constraint_object.keyframe_insert(data_path = 'location',frame = i)

        bpy.context.view_layer.update()
        cam_rotations.append(render_utils.get_3x4_RT_matrix_from_blender(cam))

    try:
        bpy.data.objects['Cube'].hide_render = True
    except:
        pass
    
    
    K = render_utils.get_calibration_matrix_K_from_blender(cam.data)
    ###########################################
    # render scene
    scn.frame_set(0)
    
    for i in np.arange(n_frames):
        scn.frame_set(i)
        bpy.ops.render.render()
    ##########################################

    render_end_time = time.time()
    
    # prepare and output metadata
    metadata = {}
    
    metadata['objects'] = []
    metadata['camera'] = {
        "poses":[],
        "K":np.array(K).tolist()
    }
    metadata["scene"] = {
        "floor_pbr":config['appearance_assets']['floor_pbr'],
        "background_hdr":config['appearance_assets']['background_hdr']
    }
    
    for i in range(n_frames):
        metadata['camera']['poses'].append(
            {
                "rotation":np.array(cam_rotations[i]).tolist(),
            }
        )

    ##########################
    # with open(os.path.join(output_dirpath, "metadata.json"), "r") as f:
    #     meta_dict = json.load(f)
    #     objects_dict = meta_dict['objects']
    #     rendering_info = meta_dict['rendering_info']
    # metadata['objects'] = objects_dict
    # metadata['rendering_info'] = rendering_info
    ##########################
    
    for obj in obj_list:
        obj = bpy.data.objects[obj]
        metadata['objects'].append(
            {
                "name":obj.name,
                "rotation_matrix":np.array(obj.matrix_world)[:3,:3].tolist(),
                "rotation_euler":np.array(obj.rotation_euler).tolist(),
                "location":np.array(obj.location).tolist(),
                "scale":np.array(obj.scale).tolist()
            }
        )
    
    metadata['rendering_info'] = {
        "object_loading_time":load_end_time - load_start_time
    }
    
    meta_str = json.dumps(metadata, indent=True)

    with open(os.path.join(output_dirpath, "metadata_updated.json"), "w") as f:
        f.write(meta_str)
    # except Exception as e:
    #     print(e)


def main1():

    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()  # Delete all objects

    # Add a cube
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

    # Add a camera and set it as the active camera
    bpy.ops.object.camera_add(location=(0, 0, 10))  # Add a camera
    camera = bpy.context.object  # Reference the newly added camera
    bpy.context.scene.camera = camera  # Set it as the active camera

    # Add a light
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 5))  # Add a light source

    # Set render settings
    bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles render engine
    bpy.context.scene.render.filepath = "/tmp/output.png"  # Render output file path
    bpy.context.scene.render.image_settings.file_format = 'PNG'  # Output format

    try:
        bpy.ops.render.render(write_still=True)
        print("Render completed. Check /tmp/output.png")
    except Exception as e:
        print(f"Render failed: {e}")

if __name__ == "__main__":
    main()