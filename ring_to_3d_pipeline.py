"""
Ring to 3D Pipeline
====================
A complete pipeline that takes a text prompt, generates a ring image,
and processes it through segmentation, depth estimation, point cloud generation,
and mesh reconstruction to create manufacturing-ready 3D models.

Pipeline Steps:
1. Generate ring image from text prompt (Stable Diffusion)
2. Segment ring components using Grounded-SAM (metal, gemstones, prongs, background)
3. Run depth estimation using Metric3D
4. Convert depth map to point cloud using camera intrinsics
5. Process point cloud (cleaning, smoothing) using Open3D and point-cloud-utils
6. Create separate meshes per component
7. Export to manufacturing formats (STL, STEP, 3DM)
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

# Import our modules
from modules.image_generator import RingImageGenerator
from modules.segmentation import GroundedSAMSegmenter
from modules.depth_estimation import DepthEstimator
from modules.point_cloud import PointCloudProcessor
from modules.mesh_reconstruction import MeshReconstructor
from modules.exporter import MeshExporter
from utils.config import PipelineConfig
from utils.logger import setup_logger


class RingTo3DPipeline:
    """
    Main pipeline class that orchestrates the entire workflow from
    text prompt to manufacturing-ready 3D meshes.
    """
    
    def __init__(self, config: PipelineConfig = None, output_dir: str = "output"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Pipeline configuration object
            output_dir: Directory to save all outputs
        """
        self.config = config or PipelineConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(self.output_dir / "pipeline.log")
        
        # Initialize all pipeline components
        self.logger.info("Initializing pipeline components...")
        
        self.image_generator = RingImageGenerator(self.config)
        self.segmenter = GroundedSAMSegmenter(self.config)
        self.depth_estimator = DepthEstimator(self.config)
        self.point_cloud_processor = PointCloudProcessor(self.config)
        self.mesh_reconstructor = MeshReconstructor(self.config)
        self.exporter = MeshExporter(self.config)
        
        self.logger.info("Pipeline initialized successfully!")
    
    def run(self, prompt: str = None, run_name: str = None, image_path: str = None) -> dict:
        """
        Run the complete pipeline from prompt or existing image to 3D meshes.
        
        Args:
            prompt: Text prompt describing the ring to generate (optional if image_path provided)
            run_name: Optional name for this run (used for output folder)
            image_path: Path to existing image (skips generation step)
            
        Returns:
            Dictionary containing all outputs and metadata
        """
        # Create run-specific output directory
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "prompt": prompt,
            "run_name": run_name,
            "output_dir": str(run_dir),
            "steps": {}
        }
        
        try:
            # Step 1: Generate or use existing ring image
            self.logger.info("=" * 50)
            
            if image_path:
                # Use existing image
                import shutil
                self.logger.info("Step 1: Using existing image...")
                self.logger.info(f"Source: {image_path}")
                
                source_path = Path(image_path)
                if not source_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                # Copy to output directory
                dest_path = run_dir / "01_input_image.png"
                shutil.copy(source_path, dest_path)
                image_path = dest_path
                
                results["steps"]["image_generation"] = {
                    "status": "skipped (using existing image)",
                    "output": str(image_path)
                }
            else:
                # Generate from prompt
                self.logger.info("Step 1: Generating ring image from prompt...")
                self.logger.info(f"Prompt: {prompt}")
                
                image_path = self.image_generator.generate(
                    prompt=prompt,
                    output_path=run_dir / "01_generated_image.png"
                )
                results["steps"]["image_generation"] = {
                    "status": "success",
                    "output": str(image_path)
                }
            
            self.logger.info(f"Image saved to: {image_path}")
            
            # Step 2: Segment ring components
            self.logger.info("=" * 50)
            self.logger.info("Step 2: Segmenting ring components...")
            
            segments = self.segmenter.segment(
                image_path=image_path,
                output_dir=run_dir / "02_segments",
                components=["ring metal body", "gemstone", "prong", "diamond"]
            )
            results["steps"]["segmentation"] = {
                "status": "success",
                "outputs": segments
            }
            self.logger.info(f"Segmented {len(segments)} components")
            
            # Step 3: Run depth estimation
            self.logger.info("=" * 50)
            self.logger.info("Step 3: Estimating depth map...")
            
            depth_map, depth_path = self.depth_estimator.estimate(
                image_path=image_path,
                output_path=run_dir / "03_depth_map.png"
            )
            results["steps"]["depth_estimation"] = {
                "status": "success",
                "output": str(depth_path)
            }
            self.logger.info(f"Depth map saved to: {depth_path}")
            
            # Step 4: Convert to point cloud
            self.logger.info("=" * 50)
            self.logger.info("Step 4: Converting depth map to point cloud...")
            
            point_clouds = self.point_cloud_processor.depth_to_pointcloud(
                depth_map=depth_map,
                segments=segments,
                output_dir=run_dir / "04_point_clouds",
                camera_intrinsics=self.config.camera_intrinsics
            )
            results["steps"]["point_cloud"] = {
                "status": "success",
                "outputs": point_clouds
            }
            self.logger.info(f"Generated {len(point_clouds)} point clouds")
            
            # Step 5: Process point clouds (clean, smooth)
            self.logger.info("=" * 50)
            self.logger.info("Step 5: Processing point clouds...")
            
            processed_clouds = self.point_cloud_processor.process(
                point_clouds=point_clouds,
                output_dir=run_dir / "05_processed_clouds"
            )
            results["steps"]["point_cloud_processing"] = {
                "status": "success",
                "outputs": processed_clouds
            }
            
            # Step 6: Reconstruct meshes
            self.logger.info("=" * 50)
            self.logger.info("Step 6: Reconstructing meshes...")
            
            meshes = self.mesh_reconstructor.reconstruct(
                point_clouds=processed_clouds,
                output_dir=run_dir / "06_meshes"
            )
            results["steps"]["mesh_reconstruction"] = {
                "status": "success",
                "outputs": meshes
            }
            self.logger.info(f"Reconstructed {len(meshes)} meshes")
            
            # Step 7: Export to manufacturing formats
            self.logger.info("=" * 50)
            self.logger.info("Step 7: Exporting to manufacturing formats...")
            
            exports = self.exporter.export(
                meshes=meshes,
                output_dir=run_dir / "07_exports",
                formats=["stl", "obj", "ply"]  # STEP and 3DM require additional libraries
            )
            results["steps"]["export"] = {
                "status": "success",
                "outputs": exports
            }
            self.logger.info(f"Exported to: {run_dir / '07_exports'}")
            
            results["status"] = "success"
            self.logger.info("=" * 50)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"All outputs saved to: {run_dir}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            self.logger.error(f"Pipeline failed with error: {e}")
            raise
        
        return results


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate 3D manufacturing-ready models from ring descriptions"
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt describing the ring (e.g., 'elegant diamond engagement ring with gold band')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save outputs (default: output)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (default: timestamp)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: cuda)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Size of generated image (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        device=args.device,
        image_size=args.image_size
    )
    
    # Initialize and run pipeline
    pipeline = RingTo3DPipeline(config=config, output_dir=args.output_dir)
    results = pipeline.run(prompt=args.prompt, run_name=args.run_name)
    
    print("\n" + "=" * 50)
    print("Pipeline Results:")
    print("=" * 50)
    print(f"Status: {results['status']}")
    print(f"Output Directory: {results['output_dir']}")
    
    if results['status'] == 'success':
        print("\nGenerated Files:")
        for step, data in results['steps'].items():
            if 'output' in data:
                print(f"  - {step}: {data['output']}")
            elif 'outputs' in data:
                print(f"  - {step}: {len(data['outputs'])} files")


if __name__ == "__main__":
    main()
