"""
Simple Run Script
==================
A simplified interface to run the Ring to 3D pipeline with a single prompt.

Usage:
    python run.py "elegant diamond engagement ring with gold band"
    python run.py "silver wedding band with sapphire stones" --output my_ring
"""

import sys
import argparse
from pathlib import Path


def run_pipeline(prompt: str = None, output_name: str = None, image_path: str = None, use_mock: bool = False):
    """
    Run the Ring to 3D pipeline with a prompt or existing image.
    
    Args:
        prompt: Text description of the ring (optional if image_path provided)
        output_name: Name for output directory
        image_path: Path to existing image (skips generation step)
        use_mock: Use mock generators for testing without GPU
    """
    from ring_to_3d_pipeline import RingTo3DPipeline
    from utils.config import PipelineConfig
    
    # Detect device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow.")
        print("For faster processing, use a GPU with CUDA support.")
    
    # Create configuration
    config = PipelineConfig(device=device)
    
    # Initialize pipeline
    pipeline = RingTo3DPipeline(config=config, output_dir="output")
    
    # Run pipeline
    print("\n" + "=" * 60)
    print("Ring to 3D Pipeline")
    print("=" * 60)
    
    if image_path:
        print(f"\nUsing existing image: {image_path}\n")
    else:
        print(f"\nPrompt: {prompt}\n")
    
    try:
        results = pipeline.run(prompt=prompt, run_name=output_name, image_path=image_path)
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"\nOutput directory: {results['output_dir']}")
        print("\nGenerated files:")
        
        for step, data in results.get('steps', {}).items():
            if isinstance(data, dict):
                if 'output' in data:
                    print(f"  - {step}: {Path(data['output']).name}")
                elif 'outputs' in data:
                    if isinstance(data['outputs'], dict):
                        for name, info in data['outputs'].items():
                            if isinstance(info, dict) and 'path' in info:
                                print(f"  - {step}/{name}: {Path(info['path']).name}")
                            elif isinstance(info, str):
                                print(f"  - {step}/{name}: {Path(info).name}")
        
        return results
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that CUDA is available if using GPU")
        print("3. Download required model checkpoints (see requirements.txt)")
        raise


def run_quick_test():
    """Run a quick test with mock generators."""
    print("Running quick test with mock generators...")
    
    from utils.config import PipelineConfig
    from modules.image_generator import MockImageGenerator
    
    config = PipelineConfig(device="cpu")
    
    # Test image generation
    generator = MockImageGenerator(config)
    output_path = Path("output/test")
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_path = generator.generate(
        prompt="test ring",
        output_path=output_path / "test_image.png"
    )
    
    print(f"Test image created: {image_path}")
    print("Quick test passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D ring models from text prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py "elegant diamond engagement ring with gold band"
  python run.py "platinum wedding band with three diamonds" --output platinum_ring
  python run.py --test  # Run quick test without GPU
        """
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        help="Text description of the ring to generate"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Name for output directory"
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to existing image (skips image generation)"
    )
    
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run quick test with mock generators"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_quick_test()
    elif args.image:
        run_pipeline(prompt=args.prompt, output_name=args.output, image_path=args.image)
    elif args.prompt:
        run_pipeline(prompt=args.prompt, output_name=args.output)
    else:
        parser.print_help()
        print("\n\nExamples:")
        print('  python run.py "elegant diamond engagement ring with gold band"')
        print('  python run.py "silver wedding band with sapphire" --output my_ring')
        print('  python run.py --image path/to/ring.png  # Use existing image')


if __name__ == "__main__":
    main()
