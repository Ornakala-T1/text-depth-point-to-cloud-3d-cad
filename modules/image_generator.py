"""
Image Generation Module
========================
Generates high-quality ring images using Stable Diffusion with
consistent camera orientation optimized for 3D reconstruction.
"""

import torch
from pathlib import Path
from PIL import Image
import numpy as np

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class RingImageGenerator:
    """
    Generates ring images using Stable Diffusion with prompts optimized
    for consistent camera orientation and 3D reconstruction quality.
    """
    
    # Prompt engineering templates for consistent ring images
    PROMPT_TEMPLATES = {
        "standard": (
            "{prompt}, "
            "professional product photography, "
            "centered ring on plain white background, "
            "top-down three-quarter view at 45 degrees, "
            "studio lighting, "
            "sharp focus, "
            "high detail, "
            "8k resolution, "
            "jewelry photography, "
            "clean background"
        ),
        "detailed": (
            "{prompt}, "
            "ultra detailed product shot, "
            "professional jewelry photography, "
            "ring centered in frame, "
            "three-quarter view angle, "
            "soft box studio lighting, "
            "white seamless background, "
            "macro photography, "
            "sharp focus on ring details, "
            "8k uhd, "
            "photorealistic"
        ),
        "cad_friendly": (
            "{prompt}, "
            "CAD render style, "
            "centered ring on neutral gray background, "
            "orthographic three-quarter view, "
            "even lighting no harsh shadows, "
            "high detail geometry, "
            "clean edges, "
            "product visualization, "
            "3D reconstruction friendly"
        )
    }
    
    NEGATIVE_PROMPT = (
        "blurry, low quality, distorted, deformed, "
        "multiple rings, hands, fingers, human, "
        "text, watermark, logo, "
        "cluttered background, busy background, "
        "motion blur, out of focus, "
        "low resolution, jpeg artifacts, "
        "oversaturated, underexposed, overexposed"
    )
    
    def __init__(self, config):
        """
        Initialize the image generator.
        
        Args:
            config: PipelineConfig object with settings
        """
        self.config = config
        self.device = config.device
        self.image_size = config.image_size
        self.pipe = None
        
    def _load_model(self):
        """Load the Stable Diffusion model."""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers library not installed. "
                "Install with: pip install diffusers transformers accelerate"
            )
        
        if self.pipe is None:
            print("Loading Stable Diffusion model...")
            
            # Use Stable Diffusion 2.1 for better quality
            model_id = self.config.sd_model_id
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable for faster inference
                requires_safety_checker=False
            )
            
            # Use DPM++ scheduler for better results
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass  # xformers not available
            
            print("Model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        output_path: str | Path,
        template: str = "detailed",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = None
    ) -> Path:
        """
        Generate a ring image from a text prompt.
        
        Args:
            prompt: Description of the ring to generate
            output_path: Path to save the generated image
            template: Prompt template to use ('standard', 'detailed', 'cad_friendly')
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            
        Returns:
            Path to the saved image
        """
        self._load_model()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the full prompt using template
        template_str = self.PROMPT_TEMPLATES.get(template, self.PROMPT_TEMPLATES["detailed"])
        full_prompt = template_str.format(prompt=prompt)
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"Generating image with prompt: {prompt}")
        
        # Generate image
        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=self.NEGATIVE_PROMPT,
            height=self.image_size,
            width=self.image_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        image = result.images[0]
        
        # Save the image
        image.save(output_path)
        print(f"Image saved to: {output_path}")
        
        return output_path
    
    def generate_multiple_views(
        self,
        prompt: str,
        output_dir: str | Path,
        num_views: int = 4,
        seed: int = 42
    ) -> list[Path]:
        """
        Generate multiple views of the ring for better 3D reconstruction.
        
        Args:
            prompt: Description of the ring
            output_dir: Directory to save images
            num_views: Number of views to generate
            seed: Base seed for reproducibility
            
        Returns:
            List of paths to generated images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        view_prompts = [
            f"{prompt}, front view",
            f"{prompt}, three-quarter view",
            f"{prompt}, side view",
            f"{prompt}, top view"
        ]
        
        paths = []
        for i, view_prompt in enumerate(view_prompts[:num_views]):
            path = self.generate(
                prompt=view_prompt,
                output_path=output_dir / f"view_{i:02d}.png",
                seed=seed + i
            )
            paths.append(path)
        
        return paths


class MockImageGenerator:
    """
    Mock image generator for testing without GPU/models.
    Creates a simple placeholder image.
    """
    
    def __init__(self, config):
        self.config = config
        self.image_size = config.image_size
    
    def generate(
        self,
        prompt: str,
        output_path: str | Path,
        **kwargs
    ) -> Path:
        """Generate a mock placeholder image."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a simple gradient image as placeholder
        img = Image.new('RGB', (self.image_size, self.image_size), color='white')
        pixels = img.load()
        
        # Draw a simple ring shape
        center = self.image_size // 2
        outer_radius = self.image_size // 3
        inner_radius = self.image_size // 4
        
        for y in range(self.image_size):
            for x in range(self.image_size):
                dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
                if inner_radius < dist < outer_radius:
                    # Gold color for ring
                    pixels[x, y] = (218, 165, 32)
        
        img.save(output_path)
        print(f"Mock image saved to: {output_path}")
        
        return output_path
