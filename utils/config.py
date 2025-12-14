"""
Pipeline Configuration
=======================
Configuration settings for the Ring to 3D pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PipelineConfig:
    """Configuration for the Ring to 3D pipeline."""
    
    # Device settings
    device: str = "cuda"
    
    # Image generation settings
    image_size: int = 512
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"  # Public model, no auth required
    
    # Grounded-SAM settings
    grounding_dino_config: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint: str = "groundingdino_swint_ogc.pth"
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    
    # Depth estimation settings
    depth_model: str = "vit_small"  # Options: vit_small, vit_large, vit_giant2
    
    # Camera intrinsics (default for product photography)
    camera_intrinsics: Dict[str, float] = field(default_factory=lambda: {
        'fx': 1000.0,
        'fy': 1000.0,
        'cx': 256.0,  # Half of image_size
        'cy': 256.0
    })
    
    # Mesh reconstruction settings
    mesh_algorithm: str = "poisson"  # Options: poisson, ball_pivoting, alpha_shape
    
    # Processing settings
    voxel_downsample: bool = True
    remove_outliers: bool = True
    smooth_mesh: bool = True
    
    def __post_init__(self):
        """Update camera intrinsics based on image size."""
        self.camera_intrinsics['cx'] = self.image_size / 2
        self.camera_intrinsics['cy'] = self.image_size / 2
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'device': self.device,
            'image_size': self.image_size,
            'sd_model_id': self.sd_model_id,
            'grounding_dino_config': self.grounding_dino_config,
            'grounding_dino_checkpoint': self.grounding_dino_checkpoint,
            'sam_checkpoint': self.sam_checkpoint,
            'sam_model_type': self.sam_model_type,
            'depth_model': self.depth_model,
            'camera_intrinsics': self.camera_intrinsics,
            'mesh_algorithm': self.mesh_algorithm,
            'voxel_downsample': self.voxel_downsample,
            'remove_outliers': self.remove_outliers,
            'smooth_mesh': self.smooth_mesh
        }


@dataclass
class RingPromptConfig:
    """Configuration for ring-specific prompts and components."""
    
    # Ring components to segment
    components: list = field(default_factory=lambda: [
        "ring metal body",
        "ring band", 
        "gemstone",
        "diamond",
        "prong",
        "setting",
        "halo"
    ])
    
    # Metal types for prompt engineering
    metal_types: list = field(default_factory=lambda: [
        "gold",
        "white gold",
        "rose gold",
        "platinum",
        "silver",
        "titanium"
    ])
    
    # Gemstone types
    gemstone_types: list = field(default_factory=lambda: [
        "diamond",
        "ruby",
        "sapphire",
        "emerald",
        "amethyst",
        "topaz"
    ])
    
    # Ring styles
    ring_styles: list = field(default_factory=lambda: [
        "solitaire",
        "halo",
        "three-stone",
        "pavé",
        "vintage",
        "modern",
        "classic"
    ])
