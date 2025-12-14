"""
Point Cloud Module
===================
Converts depth maps to point clouds using camera intrinsics.
Includes processing, cleaning, and smoothing of point clouds.

Uses point-cloud-utils (pcu) and Open3D for processing.
Reference: https://github.com/fwilliams/point-cloud-utils
"""

import numpy as np
from pathlib import Path
import cv2

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    import point_cloud_utils as pcu
    PCU_AVAILABLE = True
except ImportError:
    PCU_AVAILABLE = False


class PointCloudProcessor:
    """
    Converts depth maps to point clouds and processes them
    for mesh reconstruction.
    
    Uses the equation:
    3D_point = depth × inverse(camera_matrix) × pixel_coordinates
    """
    
    def __init__(self, config):
        """
        Initialize the point cloud processor.
        
        Args:
            config: PipelineConfig object with settings
        """
        self.config = config
        
    def depth_to_pointcloud(
        self,
        depth_map: np.ndarray,
        segments: dict,
        output_dir: str | Path,
        camera_intrinsics: dict = None,
        rgb_image: np.ndarray = None
    ) -> dict:
        """
        Convert depth map to point clouds, one per segment.
        
        Args:
            depth_map: 2D numpy array of depth values
            segments: Dictionary of segmentation masks from segmentation module
            output_dir: Directory to save point clouds
            camera_intrinsics: Camera parameters {fx, fy, cx, cy}
            rgb_image: Optional RGB image for coloring points
            
        Returns:
            Dictionary mapping segment names to point cloud file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        h, w = depth_map.shape
        
        # Default camera intrinsics (assuming standard product photography setup)
        if camera_intrinsics is None:
            camera_intrinsics = {
                'fx': 1000.0,  # Focal length x
                'fy': 1000.0,  # Focal length y
                'cx': w / 2,   # Principal point x
                'cy': h / 2    # Principal point y
            }
        
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        
        # Create pixel coordinate grids
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)
        
        # Convert to 3D coordinates
        # X = (u - cx) * depth / fx
        # Y = (v - cy) * depth / fy
        # Z = depth
        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack into point cloud [H, W, 3]
        points_full = np.stack([x, y, z], axis=-1)
        
        point_clouds = {}
        
        # Process each segment
        for name, segment in segments.items():
            if name == "background":
                continue  # Skip background
                
            mask = segment.get("mask")
            if mask is None:
                # Load mask from file
                mask_path = segment.get("mask_path")
                if mask_path:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
                else:
                    continue
            
            # Ensure mask is same size as depth
            if mask.shape != depth_map.shape:
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Extract points for this segment
            valid_mask = (mask > 0.5) & (z > 0) & np.isfinite(z)
            points = points_full[valid_mask]
            
            if len(points) < 100:
                print(f"  Skipping {name}: too few points ({len(points)})")
                continue
            
            # Get colors if RGB image provided
            colors = None
            if rgb_image is not None:
                colors = rgb_image[valid_mask] / 255.0
            
            # Save point cloud
            ply_path = output_dir / f"{name}_pointcloud.ply"
            
            if OPEN3D_AVAILABLE:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(str(ply_path), pcd)
            elif PCU_AVAILABLE:
                # Use point-cloud-utils
                if colors is not None:
                    pcu.save_mesh_vn(str(ply_path), points, colors)
                else:
                    pcu.save_mesh_v(str(ply_path), points)
            else:
                # Simple PLY writer
                self._write_ply(ply_path, points, colors)
            
            point_clouds[name] = {
                "path": str(ply_path),
                "num_points": len(points),
                "points": points,
                "colors": colors
            }
            
            print(f"  {name}: {len(points)} points saved to {ply_path}")
        
        return point_clouds
    
    def _write_ply(
        self,
        path: Path,
        points: np.ndarray,
        colors: np.ndarray = None
    ):
        """Simple PLY writer for when no libraries are available."""
        has_colors = colors is not None and len(colors) == len(points)
        
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_colors:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i, p in enumerate(points):
                line = f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
                if has_colors:
                    c = (colors[i] * 255).astype(int)
                    line += f" {c[0]} {c[1]} {c[2]}"
                f.write(line + "\n")
    
    def process(
        self,
        point_clouds: dict,
        output_dir: str | Path
    ) -> dict:
        """
        Process point clouds: remove outliers, smooth, and downsample.
        
        Args:
            point_clouds: Dictionary of point clouds from depth_to_pointcloud
            output_dir: Directory to save processed point clouds
            
        Returns:
            Dictionary of processed point cloud paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed = {}
        
        for name, pc_data in point_clouds.items():
            print(f"  Processing {name}...")
            
            points = pc_data.get("points")
            colors = pc_data.get("colors")
            
            if points is None:
                # Load from file
                path = pc_data.get("path")
                if path and OPEN3D_AVAILABLE:
                    pcd = o3d.io.read_point_cloud(path)
                    points = np.asarray(pcd.points)
                    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                else:
                    continue
            
            if len(points) < 100:
                print(f"    Skipping: too few points")
                continue
            
            # Process with Open3D if available
            if OPEN3D_AVAILABLE:
                points, colors = self._process_open3d(points, colors)
            # Or with point-cloud-utils
            elif PCU_AVAILABLE:
                points, colors = self._process_pcu(points, colors)
            else:
                points, colors = self._process_simple(points, colors)
            
            # Save processed point cloud
            ply_path = output_dir / f"{name}_processed.ply"
            
            if OPEN3D_AVAILABLE:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                
                # Estimate normals (needed for mesh reconstruction)
                if len(points) > 10:
                    try:
                        pcd.estimate_normals(
                            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                radius=0.1, max_nn=30
                            )
                        )
                        if pcd.has_normals():
                            pcd.orient_normals_consistent_tangent_plane(k=min(15, len(points)-1))
                    except Exception as e:
                        print(f"    Warning: Could not estimate normals: {e}")
                
                o3d.io.write_point_cloud(str(ply_path), pcd)
            else:
                self._write_ply(ply_path, points, colors)
            
            processed[name] = {
                "path": str(ply_path),
                "num_points": len(points),
                "points": points,
                "colors": colors
            }
            
            print(f"    Processed: {len(points)} points")
        
        return processed
    
    def _process_open3d(
        self,
        points: np.ndarray,
        colors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process point cloud using Open3D."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Statistical outlier removal
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )
        
        # Voxel downsampling
        voxel_size = self._estimate_voxel_size(points)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        # Additional radius outlier removal
        pcd, ind = pcd.remove_radius_outlier(
            nb_points=16,
            radius=voxel_size * 3
        )
        
        processed_points = np.asarray(pcd.points)
        processed_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        return processed_points, processed_colors
    
    def _process_pcu(
        self,
        points: np.ndarray,
        colors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process point cloud using point-cloud-utils."""
        # Estimate good voxel size
        voxel_size = self._estimate_voxel_size(points)
        
        # Downsample using poisson disk sampling
        if colors is not None:
            idx = pcu.downsample_point_cloud_poisson_disk(points, voxel_size * 2)
            points = points[idx]
            colors = colors[idx]
        else:
            idx = pcu.downsample_point_cloud_poisson_disk(points, voxel_size * 2)
            points = points[idx]
        
        # Deduplicate points
        points, idx_i, idx_j = pcu.deduplicate_point_cloud(points, 1e-7)
        if colors is not None:
            colors = colors[idx_i]
        
        return points, colors
    
    def _process_simple(
        self,
        points: np.ndarray,
        colors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple processing without external libraries."""
        # Simple voxel grid downsampling
        voxel_size = self._estimate_voxel_size(points)
        
        # Quantize points to voxel grid
        voxel_indices = np.floor(points / voxel_size).astype(int)
        
        # Find unique voxels
        unique_voxels, inverse = np.unique(
            voxel_indices,
            axis=0,
            return_inverse=True
        )
        
        # Average points in each voxel
        processed_points = np.zeros((len(unique_voxels), 3))
        processed_colors = None
        if colors is not None:
            processed_colors = np.zeros((len(unique_voxels), 3))
        
        counts = np.zeros(len(unique_voxels))
        
        for i, inv in enumerate(inverse):
            processed_points[inv] += points[i]
            if colors is not None:
                processed_colors[inv] += colors[i]
            counts[inv] += 1
        
        processed_points /= counts[:, np.newaxis]
        if processed_colors is not None:
            processed_colors /= counts[:, np.newaxis]
        
        # Simple outlier removal (remove points far from median)
        median = np.median(processed_points, axis=0)
        distances = np.linalg.norm(processed_points - median, axis=1)
        threshold = np.percentile(distances, 95)
        
        mask = distances < threshold
        processed_points = processed_points[mask]
        if processed_colors is not None:
            processed_colors = processed_colors[mask]
        
        return processed_points, processed_colors
    
    def _estimate_voxel_size(self, points: np.ndarray) -> float:
        """Estimate appropriate voxel size based on point cloud extent."""
        bbox = points.max(axis=0) - points.min(axis=0)
        bbox_diag = np.linalg.norm(bbox)
        
        # Target around 50000-100000 points
        target_points = 75000
        current_points = len(points)
        
        # Voxel size to achieve target density
        volume = np.prod(bbox)
        if volume > 0 and current_points > 0:
            current_density = current_points / volume
            target_density = target_points / volume
            
            if current_density > target_density:
                voxel_size = (current_points / target_points) ** (1/3) * (volume / current_points) ** (1/3)
            else:
                voxel_size = bbox_diag / 500
        else:
            voxel_size = bbox_diag / 500
        
        return max(voxel_size, 0.001)  # Minimum voxel size
    
    def estimate_normals(
        self,
        points: np.ndarray,
        k_neighbors: int = 30
    ) -> np.ndarray:
        """
        Estimate surface normals for a point cloud.
        
        Args:
            points: Nx3 array of points
            k_neighbors: Number of neighbors for normal estimation
            
        Returns:
            Nx3 array of normal vectors
        """
        if OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
            )
            pcd.orient_normals_consistent_tangent_plane(k=k_neighbors // 2)
            return np.asarray(pcd.normals)
        
        elif PCU_AVAILABLE:
            normals = pcu.estimate_point_cloud_normals_knn(points, k_neighbors)
            return normals
        
        else:
            # Simple normal estimation (not recommended for real use)
            print("Warning: Normal estimation without Open3D/PCU is limited")
            return np.zeros_like(points)
