Step 1 — Generate one good image of the ring
Using prompt engineering, ensure consistent camera orientation.


Step 2 — Use SAM (Segment Anything Model) to segment the image
SAM will identify. Use this repo (https://github.com/IDEA-Research/Grounded-Segment-Anything):
Ring metal body
Gemstones
Prongs
Background
This allows you to treat each component separately.


Step 3 — Run a Depth Estimation Model
Use this - https://github.com/YvanYin/Metric3D:
Metric3D (a known depth-estimation model)
This produces a depth map for each pixel.


Step 4 — Convert Depth Map + Camera Matrix → Point Cloud
Using the depth map and the camera intrinsics:
3D_point = depth × inverse(camera_matrix) × pixel_coordinates
This results in a point cloud representing the ring geometry.


Step 5 — Process the Point Cloud (Open3D, Noise Removal, Smoothing)
Use this -> (https://github.com/fwilliams/point-cloud-utils). Suggests using Open3D to:
Clean outliers
Smooth surfaces
Reconstruct mesh

Use this: https://github.com/lllyasviel/ControlNet

Step 6 — Create Separate Meshes per Component
Because SAM labeled segments, you can:
Mesh diamonds separately
Mesh prongs separately
Mesh the ring shank separately
This solves the manufacturing requirement: not a single unified mesh.


Step 7 — Merge Meshes → CAD/3DM → Manufacturing
Output can be exported as:
STL
STEP
3DM
This is the first time in your pipeline the result becomes manufacturing grade.