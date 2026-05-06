import argparse
import os
from pytorch_fid import fid_score
import torch
import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def calculate_fid(real_dir, fake_dir, batch_size=8, device='cuda', dims=2048):
    """
    Calculate FID score between images in real_dir and fake_dir.
    
    Args:
        real_dir (str): Path to directory containing real images
        fake_dir (str): Path to directory containing generated images
        batch_size (int): Batch size for FID calculation
        device (str): Device to run computation ('cuda' or 'cpu')
        dims (int): Dimensionality of Inception V3 features
    Returns:
        float: FID score
    """
    logger = setup_logger()
    
    # Validate directories
    if not os.path.exists(real_dir):
        raise ValueError(f"Real images directory does not exist: {real_dir}")
    if not os.path.exists(fake_dir):
        raise ValueError(f"Generated images directory does not exist: {fake_dir}")
    
    real_images = [f for f in os.listdir(real_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    fake_images = [f for f in os.listdir(fake_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not real_images or not fake_images:
        raise ValueError("No valid images found in one or both directories")
    
    logger.info(f"Found {len(real_images)} real images and {len(fake_images)} generated images")
    
    # Calculate FID
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=batch_size,
            device=device,
            dims=dims,
            num_workers=4
        )
        logger.info(f"FID score: {fid_value:.2f}")
        return fid_value
    except Exception as e:
        logger.error(f"Error calculating FID: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID score between real and generated images")
    parser.add_argument("--real_dir", type=str, required=True, help="Path to directory with real images")
    parser.add_argument("--fake_dir", type=str, required=True, help="Path to directory with generated images")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for FID calculation")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048], help="Feature dimensionality")
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if args.device == "cpu" or (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    
    # Calculate FID
    fid = calculate_fid(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        batch_size=args.batch_size,
        device=device,
        dims=args.dims
    )
    print(f"Final FID score: {fid:.2f}")