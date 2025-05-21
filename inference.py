import argparse
import os # For directory creation
import torch
import numpy as np # For saving waveforms
import yaml # For configuration loading (explicitly requested)

from utils import get_config
from networks import SeisGen_ACGAN_real_dist # Import the generator model

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate waveforms using a trained ACGAN model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file (e.g., 'configs/seismo.yaml')")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained generator model checkpoint (.pt file)")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the generated waveforms")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of waveforms to generate (default: 10)")
    # Changed 'magnitude' to 'distance'
    parser.add_argument('--distance', type=float, default=10.0, help="Desired distance for generation (e.g., in km).") 
    
    # Determine default device
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device, help=f"Device to run inference on (default: {default_device})")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("Parsed arguments:")
    print(f"  Config file: {args.config}")
    print(f"  Model path: {args.model_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of samples: {args.num_samples}")
    # Changed 'Magnitude' to 'Distance' and 'args.magnitude' to 'args.distance'
    print(f"  Distance: {args.distance}") 
    print(f"  Device: {args.device}")

    # Load configuration
    config = get_config(args.config)
    print("\nLoaded configuration:")
    print(f"  Generator Length: {config['gen_length']}")
    print(f"  Generator Noise Length: {config['gen']['noise_length']}")

    # Initialize the generator model
    print("\nInitializing generator model...")
    generator = SeisGen_ACGAN_real_dist(
        gen_length=config['gen_length'],
        params=config['gen']
    )
    print("Generator model initialized.")

    # Load trained weights
    print(f"Loading trained weights from {args.model_path}...")
    try:
        state_dict = torch.load(args.model_path, map_location=args.device)
        # Attempt to load common state dict key patterns
        if 'model_state_dict_gen' in state_dict:
            generator.load_state_dict(state_dict['model_state_dict_gen'])
        elif 'gen_state_dict' in state_dict: 
            generator.load_state_dict(state_dict['gen_state_dict'])
        elif 'generator_state_dict' in state_dict: 
            generator.load_state_dict(state_dict['generator_state_dict'])
        elif 'state_dict' in state_dict: 
            try:
                generator.load_state_dict(state_dict)
                print("Loaded state_dict directly.")
            except RuntimeError:
                if 'model' in state_dict and isinstance(state_dict['model'], dict): 
                    generator.load_state_dict(state_dict['model'])
                else: 
                    raise KeyError("Suitable generator state dictionary key not found in checkpoint after trying common patterns and nested 'model' key.")
        else: 
            generator.load_state_dict(state_dict)
            print("Loaded state_dict directly as no common key was found.")

        print("Trained weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model checkpoint file not found at {args.model_path}")
        exit(1)
    except KeyError as e:
        print(f"Error: Could not find key {e} in the model checkpoint. Common keys are 'model_state_dict_gen', 'gen_state_dict', 'generator_state_dict'. Please check your checkpoint file.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model weights: {e}")
        exit(1)


    # Set model to evaluation mode and move to device
    generator.eval()
    generator.to(args.device)
    print(f"Model set to evaluation mode and moved to {args.device}.")

    # Core inference logic
    print("\nStarting waveform generation...")
    noise_length = config['gen']['noise_length']
    
    # Generate random noise vectors
    noise = torch.randn(args.num_samples, noise_length, device=args.device)
    print(f"Generated noise of shape: {noise.shape}")

    # Prepare distance condition tensor (changed from magnitude)
    distances = torch.full((args.num_samples,), args.distance, device=args.device).float()
    print(f"Prepared distances tensor of shape: {distances.shape} with value: {args.distance}")

    # Perform inference
    with torch.no_grad():
        print("Running generator model...")
        # Changed 'magnitudes' to 'distances'
        generated_waveforms = generator(noise, distances) 
        print("Waveform generation complete.")

    # Print the shape of the generated waveforms
    print(f"Shape of generated waveforms: {generated_waveforms.shape}")

    # Save generated waveforms
    print(f"\nSaving generated waveforms to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    waveforms_np = generated_waveforms.detach().cpu().numpy()

    for i in range(args.num_samples):
        sample_waveform = waveforms_np[i]
        # Filename uses a consistent format for distance, e.g., 10.0 -> 10p0
        # Changed filename to reflect distance and format
        dist_str = f"{args.distance:.1f}".replace('.', 'p') # Format to one decimal place and replace . with p
        filename = f"waveform_sample_{i}_dist_{dist_str}.npy"
        filepath = os.path.join(args.output_dir, filename)
        np.save(filepath, sample_waveform)

    print(f"Successfully saved {args.num_samples} waveforms in {args.output_dir}")

    # Note on waveform data range and denormalization
    # The saved waveforms are in the range [0, 1] due to the generator's sigmoid output.
    # To convert them to the original physical scale, denormalization would be
    # necessary, typically using the min/max values from the original training data's
    # scaling process. This script does not perform denormalization.
    print("\nNote: Saved waveforms are normalized in the range [0, 1].")
    print("Denormalization (e.g., using training data's min/max values) is needed to convert to physical scale.")
