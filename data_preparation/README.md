# DECOR-GAN data preparation
Data preparation code for paper DECOR-GAN: 3D Shape Detailization by Conditional Refinement.

## Dependencies
Requirements:
- Python 3.x with numpy, opencv-python and cython

Build Cython module:
```
python setup.py build_ext --inplace
```

## Usage

Step 0: download ShapeNet V1 from [ShapeNet](https://www.shapenet.org/), and change the ShapeNet data directories in all python code files.

Step 1: run *1_simplify_obj.py* to normalize the shapes and remove unnecessary attributes (e.g., texture UV).
```
python 1_simplify_obj.py 03001627
```

Step 2: run *2_voxelize.py* to voxelize shapes using [binvox](https://www.patrickmin.com/binvox/).
```
python 2_voxelize.py 03001627
```
You may run *2_zoptional_render_vox.py* to visualize the voxels.
```
python 2_zoptional_render_vox.py 03001627
```

Step 3: run *3_floodfill.py*, *3_depthfusion_5views.py*, or *3_depthfusion_17views.py* to obtain voxels with 0-voxels outside and 1-voxels inside. You may choose different methods depending on what you need. *3_floodfill.py* works best when you have high-quality meshes; *3_depthfusion_5views.py* works best when you have low-quality meshes (e.g., not watertight and have holes on surface), but loses a lot of details. Note that *3_depthfusion_5views.py* and *3_depthfusion_17views.py* use z-buffer based carving method, and they are designed to ignore bottom views (since these views are usually not visible by people). Here is a list of methods used in this work on ShapeNet categories:

| Category   | Coarse shapes   | Detailed shapes |
|:---------- | ---------------:| ---------------:|
| chair      | depth_fusion_17 | floodfill       |
| table      | depth_fusion_17 | depth_fusion_17 |
| car        | depth_fusion_5  | depth_fusion_5  |
| airplane   | depth_fusion_5  | depth_fusion_5  |
| motorbike  | depth_fusion_5  | depth_fusion_5  |
| laptop     | depth_fusion_5  | depth_fusion_5  |
| plant      | depth_fusion_5  | floodfill       |

The code runs slower than the previous ones, therefore we recommend using multiple processes:
```
python 3_floodfill.py <category_id> <process_id> <total_num_of_processes>
python 3_depthfusion_5views.py <category_id> <process_id> <total_num_of_processes>
python 3_depthfusion_17views.py <category_id> <process_id> <total_num_of_processes>
```
For instance, open 4 terminals and run one of the following commands in each terminal:
```
python 3_floodfill.py 03001627 0 4
python 3_floodfill.py 03001627 1 4
python 3_floodfill.py 03001627 2 4
python 3_floodfill.py 03001627 3 4
```

Step 4: modify and run *4_validate.py* to make sure all shapes are voxelized and filled. Visualize a few shapes using marching cubes or other tools to confirm.
