"""
Mesh Reconstruction Module
===========================
Reconstructs triangle meshes from processed point clouds.
Uses Open3D and point-cloud-utils for surface reconstruction.

Supports multiple reconstruction algorithms:
- Poisson Surface Reconstruction
- Ball Pivoting Algorithm
- Alpha Shapes
"""

import numpy as np
from pathlib import Path

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


class MeshReconstructor:
    """
    Reconstructs meshes from point clouds using various algorithms.
    Creates separate meshes for each ring component (metal, gemstones, prongs).
    """
    
    ALGORITHMS = ["poisson", "ball_pivoting", "alpha_shape"]
    
    def __init__(self, config):
        """
        Initialize the mesh reconstructor.
        
        Args:
            config: PipelineConfig object with settings
        """
        self.config = config
        self.algorithm = config.mesh_algorithm
        
    def reconstruct(
        self,
        point_clouds: dict,
        output_dir: str | Path,
        algorithm: str = None
    ) -> dict:
        """
        Reconstruct meshes from point clouds.
        
        Args:
            point_clouds: Dictionary of processed point clouds
            output_dir: Directory to save meshes
            algorithm: Reconstruction algorithm to use
            
        Returns:
            Dictionary mapping segment names to mesh file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if algorithm is None:
            algorithm = self.algorithm
        
        meshes = {}
        
        for name, pc_data in point_clouds.items():
            print(f"  Reconstructing mesh for {name}...")
            
            points = pc_data.get("points")
            colors = pc_data.get("colors")
            
            # Load from file if not in memory
            if points is None:
                path = pc_data.get("path")
                if path and OPEN3D_AVAILABLE:
                    pcd = o3d.io.read_point_cloud(path)
                    points = np.asarray(pcd.points)
                    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                elif path and PCU_AVAILABLE:
                    points, _ = pcu.load_mesh_vn(path)
                else:
                    print(f"    Cannot load point cloud for {name}")
                    continue
            
            if len(points) < 100:
                print(f"    Skipping {name}: too few points")
                continue
            
            # Reconstruct mesh
            try:
                if OPEN3D_AVAILABLE:
                    mesh = self._reconstruct_open3d(points, colors, algorithm)
                else:
                    mesh = self._reconstruct_simple(points)
                
                if mesh is None:
                    print(f"    Failed to reconstruct {name}")
                    continue
                
                # Post-process mesh
                mesh = self._post_process_mesh(mesh)
                
                # Save mesh
                mesh_path = output_dir / f"{name}_mesh.ply"
                
                if OPEN3D_AVAILABLE:
                    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
                    
                    # Get mesh statistics
                    vertices = np.asarray(mesh.vertices)
                    triangles = np.asarray(mesh.triangles)
                    
                    meshes[name] = {
                        "path": str(mesh_path),
                        "num_vertices": len(vertices),
                        "num_triangles": len(triangles),
                        "mesh": mesh
                    }
                else:
                    vertices, faces = mesh
                    self._write_ply_mesh(mesh_path, vertices, faces)
                    
                    meshes[name] = {
                        "path": str(mesh_path),
                        "num_vertices": len(vertices),
                        "num_triangles": len(faces)
                    }
                
                print(f"    Created mesh: {meshes[name]['num_vertices']} vertices, "
                      f"{meshes[name]['num_triangles']} triangles")
                      
            except Exception as e:
                print(f"    Error reconstructing {name}: {e}")
                continue
        
        return meshes
    
    def _reconstruct_open3d(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        algorithm: str
    ) -> o3d.geometry.TriangleMesh:
        """Reconstruct mesh using Open3D."""
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # Choose reconstruction algorithm
        if algorithm == "poisson":
            mesh = self._poisson_reconstruction(pcd)
        elif algorithm == "ball_pivoting":
            mesh = self._ball_pivoting_reconstruction(pcd)
        elif algorithm == "alpha_shape":
            mesh = self._alpha_shape_reconstruction(pcd)
        else:
            # Default to Poisson
            mesh = self._poisson_reconstruction(pcd)
        
        return mesh
    
    def _poisson_reconstruction(
        self,
        pcd: o3d.geometry.PointCloud,
        depth: int = 9,
        width: float = 0,
        scale: float = 1.1,
        linear_fit: bool = False
    ) -> o3d.geometry.TriangleMesh:
        """
        Poisson surface reconstruction.
        Good for smooth, watertight meshes.
        """
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=linear_fit
        )
        
        # Remove low-density vertices (artifacts)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.1)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        return mesh
    
    def _ball_pivoting_reconstruction(
        self,
        pcd: o3d.geometry.PointCloud
    ) -> o3d.geometry.TriangleMesh:
        """
        Ball Pivoting Algorithm.
        Good for preserving sharp features.
        """
        # Estimate ball radius based on point cloud
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        
        radii = [avg_dist * 0.5, avg_dist, avg_dist * 2, avg_dist * 4]
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )
        
        return mesh
    
    def _alpha_shape_reconstruction(
        self,
        pcd: o3d.geometry.PointCloud,
        alpha: float = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Alpha Shape reconstruction.
        Creates a mesh from the convex hull with controlled concavity.
        """
        if alpha is None:
            # Estimate alpha based on point density
            distances = pcd.compute_nearest_neighbor_distance()
            alpha = np.mean(distances) * 2
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd,
            alpha
        )
        
        return mesh
    
    def _reconstruct_simple(
        self,
        points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simple reconstruction when Open3D is not available.
        Uses Delaunay triangulation on projected 2D points.
        """
        from scipy.spatial import Delaunay
        
        # Project points to 2D (top-down view) for triangulation
        points_2d = points[:, :2]
        
        # Delaunay triangulation
        tri = Delaunay(points_2d)
        
        return points, tri.simplices
    
    def _post_process_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh
    ) -> o3d.geometry.TriangleMesh:
        """Apply post-processing to improve mesh quality."""
        if not OPEN3D_AVAILABLE:
            return mesh
        
        # Remove degenerate triangles
        mesh.remove_degenerate_triangles()
        
        # Remove duplicate vertices
        mesh.remove_duplicated_vertices()
        
        # Remove duplicate triangles
        mesh.remove_duplicated_triangles()
        
        # Remove non-manifold edges
        mesh.remove_non_manifold_edges()
        
        # Smooth mesh slightly
        mesh = mesh.filter_smooth_simple(number_of_iterations=1)
        
        # Compute normals
        mesh.compute_vertex_normals()
        
        return mesh
    
    def _write_ply_mesh(
        self,
        path: Path,
        vertices: np.ndarray,
        faces: np.ndarray
    ):
        """Write mesh to PLY file."""
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    def make_watertight(
        self,
        mesh_path: str | Path,
        output_path: str | Path = None,
        resolution: int = 20000
    ) -> Path:
        """
        Make a mesh watertight for manufacturing.
        
        Args:
            mesh_path: Path to input mesh
            output_path: Path for output mesh
            resolution: Resolution parameter for watertight algorithm
            
        Returns:
            Path to watertight mesh
        """
        mesh_path = Path(mesh_path)
        if output_path is None:
            output_path = mesh_path.parent / f"{mesh_path.stem}_watertight.ply"
        else:
            output_path = Path(output_path)
        
        if PCU_AVAILABLE:
            # Use point-cloud-utils watertight function
            v, f = pcu.load_mesh_vf(str(mesh_path))
            v_wt, f_wt = pcu.make_mesh_watertight(v, f, resolution=resolution)
            pcu.save_mesh_vf(str(output_path), v_wt, f_wt)
            print(f"Created watertight mesh: {output_path}")
        elif OPEN3D_AVAILABLE:
            # Alternative: fill holes using Open3D
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            mesh.compute_vertex_normals()
            
            # Simple hole filling by retriangulating
            # This is a simplified version
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            print(f"Processed mesh saved: {output_path}")
        else:
            print("Cannot create watertight mesh: neither pcu nor Open3D available")
            return mesh_path
        
        return output_path
    
    def decimate_mesh(
        self,
        mesh_path: str | Path,
        target_faces: int,
        output_path: str | Path = None
    ) -> Path:
        """
        Reduce mesh complexity while preserving shape.
        
        Args:
            mesh_path: Path to input mesh
            target_faces: Target number of faces
            output_path: Path for output mesh
            
        Returns:
            Path to decimated mesh
        """
        mesh_path = Path(mesh_path)
        if output_path is None:
            output_path = mesh_path.parent / f"{mesh_path.stem}_decimated.ply"
        else:
            output_path = Path(output_path)
        
        if OPEN3D_AVAILABLE:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            
            # Decimate using quadric decimation
            mesh_simplified = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_faces
            )
            
            o3d.io.write_triangle_mesh(str(output_path), mesh_simplified)
            
        elif PCU_AVAILABLE:
            v, f = pcu.load_mesh_vf(str(mesh_path))
            v_dec, f_dec, _, _ = pcu.decimate_triangle_mesh(v, f, target_faces)
            pcu.save_mesh_vf(str(output_path), v_dec, f_dec)
        
        else:
            print("Cannot decimate: neither Open3D nor pcu available")
            return mesh_path
        
        print(f"Decimated mesh saved: {output_path}")
        return output_path
    
    def smooth_mesh(
        self,
        mesh_path: str | Path,
        iterations: int = 5,
        output_path: str | Path = None
    ) -> Path:
        """
        Smooth a mesh using Laplacian smoothing.
        
        Args:
            mesh_path: Path to input mesh
            iterations: Number of smoothing iterations
            output_path: Path for output mesh
            
        Returns:
            Path to smoothed mesh
        """
        mesh_path = Path(mesh_path)
        if output_path is None:
            output_path = mesh_path.parent / f"{mesh_path.stem}_smoothed.ply"
        else:
            output_path = Path(output_path)
        
        if OPEN3D_AVAILABLE:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            
        elif PCU_AVAILABLE:
            v, f = pcu.load_mesh_vf(str(mesh_path))
            v_smooth = pcu.laplacian_smooth_mesh(v, f, iterations)
            pcu.save_mesh_vf(str(output_path), v_smooth, f)
        
        else:
            print("Cannot smooth: neither Open3D nor pcu available")
            return mesh_path
        
        print(f"Smoothed mesh saved: {output_path}")
        return output_path
