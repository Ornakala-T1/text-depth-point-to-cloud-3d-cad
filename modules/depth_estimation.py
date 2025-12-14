"""
Depth Estimation Module
========================
Estimates metric depth from a single image using Metric3D.
Produces depth maps suitable for 3D reconstruction.

Reference: https://github.com/YvanYin/Metric3D
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DepthEstimator:
    """
    Estimates metric depth from images using Metric3D.
    
    Metric3D is a strong foundation model for zero-shot metric depth
    and surface normal estimation from a single image.
    """
    
    # Available Metric3D models
    AVAILABLE_MODELS = {
        "vit_small": "metric3d_vit_small",
        "vit_large": "metric3d_vit_large", 
        "vit_giant2": "metric3d_vit_giant2",
        "convnext_tiny": "metric3d_convnext_tiny",
        "convnext_large": "metric3d_convnext_large"
    }
    
    def __init__(self, config):
        """
        Initialize the depth estimator.
        
        Args:
            config: PipelineConfig object with settings
        """
        self.config = config
        self.device = config.device
        self.model = None
        self.model_name = config.depth_model
        
    def _load_model(self):
        """Load the Metric3D model."""
        if self.model is not None:
            return
            
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for depth estimation")
        
        print(f"Loading Metric3D model ({self.model_name})...")
        
        try:
            # Load via PyTorch Hub
            model_hub_name = self.AVAILABLE_MODELS.get(
                self.model_name, 
                "metric3d_vit_small"
            )
            
            self.model = torch.hub.load(
                'yvanyin/metric3d',
                model_hub_name,
                pretrain=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Metric3D model loaded successfully!")
            
        except Exception as e:
            print(f"Could not load Metric3D from torch hub: {e}")
            print("Using fallback depth estimation (MiDaS)...")
            self._load_midas_fallback()
    
    def _load_midas_fallback(self):
        """Load MiDaS as fallback depth estimator."""
        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                "DPT_Large",
                pretrained=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform
            self._using_midas = True
            
            print("MiDaS fallback loaded successfully")
            
        except Exception as e:
            print(f"Could not load MiDaS fallback: {e}")
            self.model = None
            self._using_midas = False
    
    def estimate(
        self,
        image_path: str | Path,
        output_path: str | Path = None,
        return_normal: bool = False
    ) -> tuple[np.ndarray, Path]:
        """
        Estimate depth from an image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the depth map visualization
            return_normal: Whether to also estimate surface normals
            
        Returns:
            Tuple of (depth_map as numpy array, path to saved visualization)
        """
        self._load_model()
        
        image_path = Path(image_path)
        
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_depth.png"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        if self.model is None:
            # Use simple fallback
            print("Using simple depth estimation fallback")
            depth_map = self._estimate_simple_fallback(image_np)
        elif hasattr(self, '_using_midas') and self._using_midas:
            depth_map = self._estimate_midas(image_np)
        else:
            depth_map, normal_map = self._estimate_metric3d(image_np)
            
            if return_normal and normal_map is not None:
                normal_path = output_path.parent / f"{output_path.stem}_normal.png"
                self._save_normal_map(normal_map, normal_path)
        
        # Save depth visualization
        self._save_depth_visualization(depth_map, output_path)
        
        # Also save raw depth as numpy file
        raw_path = output_path.parent / f"{output_path.stem}_raw.npy"
        np.save(raw_path, depth_map)
        
        print(f"Depth map saved to: {output_path}")
        print(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
        
        return depth_map, output_path
    
    def _estimate_metric3d(self, image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimate depth using Metric3D."""
        import torch
        
        # Preprocess
        rgb = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
        rgb = rgb / 255.0
        rgb = rgb.to(self.device)
        
        # Resize to model input size
        input_size = (512, 960)  # Metric3D canonical size
        rgb_resized = F.interpolate(
            rgb,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Inference
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.model.inference({
                'input': rgb_resized
            })
        
        # Get depth map
        depth_map = pred_depth.squeeze().cpu().numpy()
        
        # Resize back to original size
        depth_map = cv2.resize(
            depth_map,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Get normal map if available
        normal_map = None
        if 'prediction_normal' in output_dict:
            normal = output_dict['prediction_normal'][:, :3, :, :]
            normal_map = normal.squeeze().permute(1, 2, 0).cpu().numpy()
            normal_map = cv2.resize(
                normal_map,
                (image_np.shape[1], image_np.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        return depth_map, normal_map
    
    def _estimate_midas(self, image_np: np.ndarray) -> np.ndarray:
        """Estimate depth using MiDaS."""
        import torch
        
        # Preprocess
        input_batch = self.transform(image_np).to(self.device)
        
        # Inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # MiDaS outputs inverse depth, convert to depth
        depth_map = 1.0 / (depth_map + 1e-6)
        
        # Normalize to reasonable range
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        depth_map = depth_map * 10  # Scale to ~10 units max
        
        return depth_map
    
    def _estimate_simple_fallback(self, image_np: np.ndarray) -> np.ndarray:
        """
        Simple depth estimation fallback using edge detection.
        This is a very rough approximation and should only be used
        when no proper depth model is available.
        """
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Use edges as proxy for depth discontinuities
        edges = cv2.Canny(gray, 50, 150)
        
        # Invert and blur to create pseudo-depth
        depth = cv2.GaussianBlur(255 - gray, (21, 21), 0)
        
        # Normalize
        depth = depth.astype(np.float32) / 255.0
        
        # Add some depth variation based on brightness
        # (brighter areas typically appear closer)
        brightness = gray.astype(np.float32) / 255.0
        depth = depth * 0.7 + brightness * 0.3
        
        # Scale to reasonable range
        depth = depth * 5.0  # 0-5 units
        
        return depth
    
    def _save_depth_visualization(
        self,
        depth_map: np.ndarray,
        output_path: Path
    ):
        """Save depth map as a colorized visualization."""
        # Normalize for visualization
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
        
        cv2.imwrite(str(output_path), depth_colored)
    
    def _save_normal_map(self, normal_map: np.ndarray, output_path: Path):
        """Save surface normal map as visualization."""
        # Normalize normals to [0, 255] for visualization
        normal_viz = ((normal_map + 1) / 2 * 255).astype(np.uint8)
        normal_viz = cv2.cvtColor(normal_viz, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(output_path), normal_viz)
        print(f"Normal map saved to: {output_path}")


class DepthFromStereo:
    """
    Alternative depth estimation using stereo reconstruction
    if multiple views are available.
    """
    
    def __init__(self, config):
        self.config = config
    
    def estimate_from_stereo(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        baseline: float = 0.1,
        focal_length: float = 1000.0
    ) -> np.ndarray:
        """
        Estimate depth from stereo image pair.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            baseline: Distance between cameras
            focal_length: Camera focal length in pixels
            
        Returns:
            Depth map
        """
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
        
        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Convert disparity to depth
        # depth = baseline * focal_length / disparity
        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = (baseline * focal_length) / disparity[valid]
        
        return depth
