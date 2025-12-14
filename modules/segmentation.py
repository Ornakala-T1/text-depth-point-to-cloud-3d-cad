"""
Segmentation Module
====================
Segments ring components using Grounded-SAM (Grounding DINO + Segment Anything).
Identifies: ring metal body, gemstones, prongs, and background.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GroundedSAMSegmenter:
    """
    Segments ring components using Grounded-SAM.
    
    Uses Grounding DINO for text-based object detection and
    Segment Anything Model (SAM) for precise segmentation.
    
    Reference: https://github.com/IDEA-Research/Grounded-Segment-Anything
    """
    
    # Default component labels for ring segmentation
    DEFAULT_COMPONENTS = [
        "ring metal body",
        "ring band",
        "gemstone",
        "diamond",
        "prong",
        "setting"
    ]
    
    def __init__(self, config):
        """
        Initialize the segmenter.
        
        Args:
            config: PipelineConfig object with settings
        """
        self.config = config
        self.device = config.device
        self.grounding_dino = None
        self.sam_predictor = None
        self._models_loaded = False
        
    def _load_models(self):
        """Load Grounding DINO and SAM models."""
        if self._models_loaded:
            return
            
        print("Loading Grounded-SAM models...")
        
        try:
            # Try loading from transformers (HuggingFace)
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            # Load Grounding DINO
            self.processor = AutoProcessor.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            )
            self.grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            ).to(self.device)
            
            print("Grounding DINO loaded from HuggingFace")
            
        except Exception as e:
            print(f"Could not load Grounding DINO from HuggingFace: {e}")
            print("Using alternative loading method...")
            self._load_models_alternative()
            
        try:
            # Load SAM
            from segment_anything import sam_model_registry, SamPredictor
            
            sam_checkpoint = self.config.sam_checkpoint
            model_type = self.config.sam_model_type
            
            if os.path.exists(sam_checkpoint):
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device=self.device)
                self.sam_predictor = SamPredictor(sam)
                print("SAM loaded successfully")
            else:
                print(f"SAM checkpoint not found at {sam_checkpoint}")
                print("Please download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
                self.sam_predictor = None
                
        except ImportError:
            print("segment_anything not installed. Using fallback segmentation.")
            self.sam_predictor = None
            
        self._models_loaded = True
        
    def _load_models_alternative(self):
        """Alternative method to load models using GroundingDINO directly."""
        try:
            # Try importing from local GroundingDINO installation
            import groundingdino
            from groundingdino.util.inference import load_model, predict
            
            config_path = self.config.grounding_dino_config
            checkpoint_path = self.config.grounding_dino_checkpoint
            
            if os.path.exists(config_path) and os.path.exists(checkpoint_path):
                self.grounding_dino = load_model(config_path, checkpoint_path)
                print("Grounding DINO loaded from local checkpoint")
            else:
                print("Grounding DINO checkpoints not found")
                self.grounding_dino = None
                
        except ImportError:
            print("GroundingDINO package not installed")
            self.grounding_dino = None
    
    def segment(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        components: list[str] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25
    ) -> dict:
        """
        Segment ring components from an image.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save segmentation masks
            components: List of components to detect (default: ring parts)
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            
        Returns:
            Dictionary mapping component names to mask file paths
        """
        self._load_models()
        
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if components is None:
            components = self.DEFAULT_COMPONENTS
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Build text prompt for Grounding DINO
        text_prompt = " . ".join(components) + " ."
        
        segments = {}
        
        if self.grounding_dino is not None and hasattr(self, 'processor'):
            # Use HuggingFace version
            segments = self._segment_with_hf(
                image, image_np, text_prompt, components,
                output_dir, box_threshold, text_threshold
            )
        else:
            # Use fallback segmentation
            print("Using fallback segmentation (color-based)")
            segments = self._segment_fallback(image_np, output_dir)
        
        # Save combined visualization
        self._save_visualization(image_np, segments, output_dir / "segmentation_viz.png")
        
        return segments
    
    def _segment_with_hf(
        self,
        image: Image.Image,
        image_np: np.ndarray,
        text_prompt: str,
        components: list[str],
        output_dir: Path,
        box_threshold: float,
        text_threshold: float
    ) -> dict:
        """Segment using HuggingFace Grounding DINO."""
        import torch
        
        # Process image
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.grounding_dino(**inputs)
        
        # Post-process results - API changed in newer versions
        try:
            # New API
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]]
            )[0]
        except TypeError:
            # Try alternative API
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                target_sizes=[image.size[::-1]]
            )[0]
        
        segments = {}
        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"]
        scores = results["scores"].cpu().numpy()
        
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            # Create mask from bounding box
            # In production, you'd use SAM for precise segmentation
            mask = self._box_to_mask(box, image_np.shape[:2])
            
            # Refine with SAM if available
            if self.sam_predictor is not None:
                mask = self._refine_with_sam(image_np, box, mask)
            
            # Save mask
            component_name = label.replace(" ", "_")
            mask_path = output_dir / f"{component_name}_{i:02d}_mask.png"
            cv2.imwrite(str(mask_path), mask * 255)
            
            segments[f"{component_name}_{i:02d}"] = {
                "mask_path": str(mask_path),
                "mask": mask,
                "bbox": box.tolist(),
                "score": float(score),
                "label": label
            }
            
            print(f"  Detected: {label} (score: {score:.2f})")
        
        return segments
    
    def _box_to_mask(self, box: np.ndarray, image_shape: tuple) -> np.ndarray:
        """Convert bounding box to binary mask."""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        mask[y1:y2, x1:x2] = 1
        return mask
    
    def _refine_with_sam(
        self,
        image_np: np.ndarray,
        box: np.ndarray,
        initial_mask: np.ndarray
    ) -> np.ndarray:
        """Refine mask using SAM."""
        self.sam_predictor.set_image(image_np)
        
        # Convert box format
        input_box = np.array(box)
        
        masks, scores, _ = self.sam_predictor.predict(
            box=input_box,
            multimask_output=True
        )
        
        # Return the mask with highest score
        best_mask = masks[np.argmax(scores)]
        return best_mask.astype(np.uint8)
    
    def _segment_fallback(self, image_np: np.ndarray, output_dir: Path) -> dict:
        """
        Fallback segmentation using color-based methods.
        Used when Grounded-SAM is not available.
        """
        segments = {}
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Segment metal (typically gold/silver - high saturation yellow or low saturation gray)
        # Gold: HSV yellow range with high saturation
        gold_lower = np.array([15, 100, 100])
        gold_upper = np.array([35, 255, 255])
        gold_mask = cv2.inRange(hsv, gold_lower, gold_upper)
        
        # Silver: low saturation, high value
        silver_lower = np.array([0, 0, 180])
        silver_upper = np.array([180, 30, 255])
        silver_mask = cv2.inRange(hsv, silver_lower, silver_upper)
        
        metal_mask = cv2.bitwise_or(gold_mask, silver_mask)
        metal_mask = self._clean_mask(metal_mask)
        
        if metal_mask.sum() > 0:
            mask_path = output_dir / "ring_metal_mask.png"
            cv2.imwrite(str(mask_path), metal_mask)
            segments["ring_metal"] = {
                "mask_path": str(mask_path),
                "mask": metal_mask // 255,
                "label": "ring metal body"
            }
        
        # Segment gemstones (typically bright, high saturation colors or clear/white)
        # Diamond: very bright areas
        _, bright_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        bright_mask = self._clean_mask(bright_mask, kernel_size=3)
        
        # Colored gems: high saturation
        gem_lower = np.array([0, 150, 100])
        gem_upper = np.array([180, 255, 255])
        gem_mask = cv2.inRange(hsv, gem_lower, gem_upper)
        
        gemstone_mask = cv2.bitwise_or(bright_mask, gem_mask)
        gemstone_mask = self._clean_mask(gemstone_mask, kernel_size=3)
        
        if gemstone_mask.sum() > 0:
            mask_path = output_dir / "gemstone_mask.png"
            cv2.imwrite(str(mask_path), gemstone_mask)
            segments["gemstone"] = {
                "mask_path": str(mask_path),
                "mask": gemstone_mask // 255,
                "label": "gemstone"
            }
        
        # Background (everything else)
        combined = cv2.bitwise_or(metal_mask, gemstone_mask)
        background_mask = cv2.bitwise_not(combined)
        
        mask_path = output_dir / "background_mask.png"
        cv2.imwrite(str(mask_path), background_mask)
        segments["background"] = {
            "mask_path": str(mask_path),
            "mask": background_mask // 255,
            "label": "background"
        }
        
        print(f"  Fallback segmentation found {len(segments)} regions")
        
        return segments
    
    def _clean_mask(
        self,
        mask: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """Clean up a mask using morphological operations."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _save_visualization(
        self,
        image_np: np.ndarray,
        segments: dict,
        output_path: Path
    ):
        """Save a visualization of all segments overlaid on the image."""
        viz = image_np.copy()
        
        # Color palette for different segments
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        for i, (name, segment) in enumerate(segments.items()):
            if name == "background":
                continue
                
            mask = segment.get("mask")
            if mask is None:
                continue
                
            color = colors[i % len(colors)]
            
            # Create colored overlay
            overlay = np.zeros_like(viz)
            overlay[mask > 0] = color
            
            # Blend with original
            alpha = 0.4
            viz = cv2.addWeighted(viz, 1, overlay, alpha, 0)
            
            # Draw label
            if "bbox" in segment:
                box = segment["bbox"]
                cv2.rectangle(
                    viz,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color, 2
                )
        
        cv2.imwrite(str(output_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        print(f"Visualization saved to: {output_path}")
