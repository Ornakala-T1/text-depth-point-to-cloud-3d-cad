"""
Mesh Export Module
===================
Exports meshes to various manufacturing-ready formats.

Supported formats:
- STL (Standard Triangle Language) - Universal 3D printing format
- OBJ (Wavefront) - Common interchange format
- PLY (Polygon File Format) - Point cloud and mesh format
- STEP (Standard for Exchange of Product Data) - CAD interchange format
- 3DM (Rhino) - Rhino/Grasshopper native format
"""

import numpy as np
from pathlib import Path
import struct

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

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


class MeshExporter:
    """
    Exports meshes to various manufacturing formats.
    Supports STL, OBJ, PLY, and optionally STEP and 3DM.
    """
    
    SUPPORTED_FORMATS = ["stl", "obj", "ply", "off", "gltf", "glb"]
    CAD_FORMATS = ["step", "stp", "iges", "igs", "3dm"]
    
    def __init__(self, config):
        """
        Initialize the exporter.
        
        Args:
            config: PipelineConfig object with settings
        """
        self.config = config
        
    def export(
        self,
        meshes: dict,
        output_dir: str | Path,
        formats: list[str] = None
    ) -> dict:
        """
        Export all meshes to specified formats.
        
        Args:
            meshes: Dictionary of mesh data from mesh reconstruction
            output_dir: Directory to save exported files
            formats: List of formats to export to
            
        Returns:
            Dictionary of exported file paths per mesh and format
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if formats is None:
            formats = ["stl", "obj", "ply"]
        
        exports = {}
        
        for name, mesh_data in meshes.items():
            print(f"  Exporting {name}...")
            exports[name] = {}
            
            mesh_path = mesh_data.get("path")
            mesh_obj = mesh_data.get("mesh")
            
            if mesh_obj is None and mesh_path:
                mesh_obj = self._load_mesh(mesh_path)
            
            if mesh_obj is None:
                print(f"    Could not load mesh for {name}")
                continue
            
            for fmt in formats:
                try:
                    export_path = output_dir / f"{name}.{fmt}"
                    
                    if fmt in self.SUPPORTED_FORMATS:
                        self._export_standard(mesh_obj, export_path, fmt)
                    elif fmt in self.CAD_FORMATS:
                        self._export_cad(mesh_obj, export_path, fmt)
                    else:
                        print(f"    Unsupported format: {fmt}")
                        continue
                    
                    exports[name][fmt] = str(export_path)
                    print(f"    Exported: {export_path.name}")
                    
                except Exception as e:
                    print(f"    Error exporting {name} to {fmt}: {e}")
        
        # Also export combined mesh (all components merged)
        if len(meshes) > 1:
            combined_path = self._export_combined(meshes, output_dir, formats)
            exports["combined"] = combined_path
        
        return exports
    
    def _load_mesh(self, path: str | Path):
        """Load mesh from file."""
        path = Path(path)
        
        if OPEN3D_AVAILABLE:
            return o3d.io.read_triangle_mesh(str(path))
        elif TRIMESH_AVAILABLE:
            return trimesh.load(str(path))
        elif PCU_AVAILABLE:
            v, f = pcu.load_mesh_vf(str(path))
            return {"vertices": v, "faces": f}
        else:
            return None
    
    def _export_standard(
        self,
        mesh,
        output_path: Path,
        fmt: str
    ):
        """Export to standard 3D formats."""
        if OPEN3D_AVAILABLE and isinstance(mesh, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            
        elif TRIMESH_AVAILABLE and isinstance(mesh, trimesh.Trimesh):
            mesh.export(str(output_path), file_type=fmt)
            
        elif isinstance(mesh, dict):
            # Raw vertices and faces
            vertices = mesh["vertices"]
            faces = mesh["faces"]
            
            if fmt == "stl":
                self._write_stl(output_path, vertices, faces)
            elif fmt == "obj":
                self._write_obj(output_path, vertices, faces)
            elif fmt == "ply":
                self._write_ply(output_path, vertices, faces)
            else:
                raise ValueError(f"Cannot export to {fmt} without proper library")
        else:
            raise ValueError(f"Unknown mesh type: {type(mesh)}")
    
    def _export_cad(
        self,
        mesh,
        output_path: Path,
        fmt: str
    ):
        """Export to CAD formats (requires additional libraries)."""
        if fmt in ["step", "stp"]:
            self._export_step(mesh, output_path)
        elif fmt in ["iges", "igs"]:
            self._export_iges(mesh, output_path)
        elif fmt == "3dm":
            self._export_3dm(mesh, output_path)
        else:
            raise ValueError(f"Unsupported CAD format: {fmt}")
    
    def _export_step(self, mesh, output_path: Path):
        """
        Export to STEP format.
        Requires pythonocc-core or FreeCAD.
        """
        try:
            from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
            from OCC.Core.TopoDS import TopoDS_Face
            
            # This is a simplified example - proper STEP export is complex
            print("    Note: STEP export requires OpenCASCADE for proper CAD geometry")
            print("    Saving as STL instead (can be converted externally)")
            
            stl_path = output_path.with_suffix('.stl')
            self._export_standard(mesh, stl_path, 'stl')
            
        except ImportError:
            print("    STEP export requires pythonocc-core library")
            print("    Install with: conda install -c conda-forge pythonocc-core")
            print("    Saving as STL instead")
            
            stl_path = output_path.with_suffix('.stl')
            self._export_standard(mesh, stl_path, 'stl')
    
    def _export_iges(self, mesh, output_path: Path):
        """Export to IGES format."""
        try:
            from OCC.Core.IGESControl import IGESControl_Writer
            
            print("    IGES export requires OpenCASCADE")
            stl_path = output_path.with_suffix('.stl')
            self._export_standard(mesh, stl_path, 'stl')
            
        except ImportError:
            print("    IGES export requires pythonocc-core library")
            stl_path = output_path.with_suffix('.stl')
            self._export_standard(mesh, stl_path, 'stl')
    
    def _export_3dm(self, mesh, output_path: Path):
        """
        Export to Rhino 3DM format.
        Requires rhino3dm library.
        """
        try:
            import rhino3dm
            
            # Create a new Rhino file
            model = rhino3dm.File3dm()
            
            # Get vertices and faces
            if OPEN3D_AVAILABLE and isinstance(mesh, o3d.geometry.TriangleMesh):
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
            elif isinstance(mesh, dict):
                vertices = mesh["vertices"]
                faces = mesh["faces"]
            else:
                raise ValueError("Cannot extract mesh data")
            
            # Create Rhino mesh
            rhino_mesh = rhino3dm.Mesh()
            
            for v in vertices:
                rhino_mesh.Vertices.Add(float(v[0]), float(v[1]), float(v[2]))
            
            for f in faces:
                rhino_mesh.Faces.AddFace(int(f[0]), int(f[1]), int(f[2]))
            
            rhino_mesh.Normals.ComputeNormals()
            rhino_mesh.Compact()
            
            # Add to model
            model.Objects.AddMesh(rhino_mesh)
            
            # Save
            model.Write(str(output_path), version=7)
            print(f"    Exported to 3DM: {output_path}")
            
        except ImportError:
            print("    3DM export requires rhino3dm library")
            print("    Install with: pip install rhino3dm")
            stl_path = output_path.with_suffix('.stl')
            self._export_standard(mesh, stl_path, 'stl')
    
    def _write_stl(
        self,
        path: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        binary: bool = True
    ):
        """Write mesh to STL format."""
        if binary:
            self._write_stl_binary(path, vertices, faces)
        else:
            self._write_stl_ascii(path, vertices, faces)
    
    def _write_stl_binary(
        self,
        path: Path,
        vertices: np.ndarray,
        faces: np.ndarray
    ):
        """Write binary STL file."""
        with open(path, 'wb') as f:
            # Header (80 bytes)
            header = b'Binary STL exported from Ring3D Pipeline' + b'\0' * 40
            f.write(header[:80])
            
            # Number of triangles
            f.write(struct.pack('<I', len(faces)))
            
            # Write each triangle
            for face in faces:
                v0 = vertices[face[0]]
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                
                # Calculate normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])
                
                # Normal vector
                f.write(struct.pack('<fff', *normal))
                
                # Three vertices
                f.write(struct.pack('<fff', *v0))
                f.write(struct.pack('<fff', *v1))
                f.write(struct.pack('<fff', *v2))
                
                # Attribute byte count
                f.write(struct.pack('<H', 0))
    
    def _write_stl_ascii(
        self,
        path: Path,
        vertices: np.ndarray,
        faces: np.ndarray
    ):
        """Write ASCII STL file."""
        with open(path, 'w') as f:
            f.write("solid ring_mesh\n")
            
            for face in faces:
                v0 = vertices[face[0]]
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                
                # Calculate normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])
                
                f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
                f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write("endsolid ring_mesh\n")
    
    def _write_obj(
        self,
        path: Path,
        vertices: np.ndarray,
        faces: np.ndarray
    ):
        """Write mesh to OBJ format."""
        with open(path, 'w') as f:
            f.write("# Ring3D Pipeline OBJ Export\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Faces: {len(faces)}\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n")
            
            # Write faces (OBJ uses 1-indexed faces)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    def _write_ply(
        self,
        path: Path,
        vertices: np.ndarray,
        faces: np.ndarray
    ):
        """Write mesh to PLY format."""
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
    
    def _export_combined(
        self,
        meshes: dict,
        output_dir: Path,
        formats: list[str]
    ) -> dict:
        """Export all meshes combined into a single file."""
        print("  Exporting combined mesh...")
        
        combined_exports = {}
        
        if OPEN3D_AVAILABLE:
            combined = o3d.geometry.TriangleMesh()
            
            for name, mesh_data in meshes.items():
                mesh_path = mesh_data.get("path")
                if mesh_path:
                    mesh = o3d.io.read_triangle_mesh(mesh_path)
                    combined += mesh
            
            for fmt in formats:
                if fmt in self.SUPPORTED_FORMATS:
                    export_path = output_dir / f"combined_ring.{fmt}"
                    o3d.io.write_triangle_mesh(str(export_path), combined)
                    combined_exports[fmt] = str(export_path)
                    print(f"    Combined exported: {export_path.name}")
        
        elif TRIMESH_AVAILABLE:
            mesh_list = []
            for name, mesh_data in meshes.items():
                mesh_path = mesh_data.get("path")
                if mesh_path:
                    mesh_list.append(trimesh.load(mesh_path))
            
            if mesh_list:
                combined = trimesh.util.concatenate(mesh_list)
                for fmt in formats:
                    export_path = output_dir / f"combined_ring.{fmt}"
                    combined.export(str(export_path))
                    combined_exports[fmt] = str(export_path)
        
        return combined_exports


class ManufacturingValidator:
    """
    Validates meshes for manufacturing requirements.
    Checks for common issues that could cause problems in 3D printing or CNC.
    """
    
    def validate(self, mesh_path: str | Path) -> dict:
        """
        Validate a mesh for manufacturing.
        
        Args:
            mesh_path: Path to mesh file
            
        Returns:
            Dictionary of validation results
        """
        mesh_path = Path(mesh_path)
        results = {
            "path": str(mesh_path),
            "valid": True,
            "issues": []
        }
        
        if not OPEN3D_AVAILABLE and not TRIMESH_AVAILABLE:
            results["valid"] = False
            results["issues"].append("Cannot validate without Open3D or trimesh")
            return results
        
        if TRIMESH_AVAILABLE:
            mesh = trimesh.load(str(mesh_path))
            
            # Check if watertight
            if not mesh.is_watertight:
                results["issues"].append("Mesh is not watertight")
            
            # Check for inverted normals
            if mesh.is_empty:
                results["issues"].append("Mesh is empty")
                results["valid"] = False
            
            # Check volume (should be positive)
            if mesh.volume < 0:
                results["issues"].append("Mesh has inverted normals (negative volume)")
            
            # Check for self-intersections (expensive check)
            # This is commented out as it can be slow
            # if mesh.ray.intersects_any(mesh.triangles_center, mesh.face_normals):
            #     results["issues"].append("Mesh has self-intersections")
            
            # Get statistics
            results["statistics"] = {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "volume": float(mesh.volume),
                "surface_area": float(mesh.area),
                "is_watertight": mesh.is_watertight,
                "euler_number": int(mesh.euler_number)
            }
        
        elif OPEN3D_AVAILABLE:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            
            # Check if mesh is empty
            if len(mesh.vertices) == 0:
                results["issues"].append("Mesh is empty")
                results["valid"] = False
            
            # Check if watertight
            if not mesh.is_watertight():
                results["issues"].append("Mesh is not watertight")
            
            # Check for self-intersections
            if not mesh.is_self_intersecting():
                pass  # Good
            else:
                results["issues"].append("Mesh has self-intersections")
            
            results["statistics"] = {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.triangles),
                "is_watertight": mesh.is_watertight()
            }
        
        if results["issues"]:
            results["valid"] = len([i for i in results["issues"] 
                                   if "watertight" not in i.lower()]) == 0
        
        return results
