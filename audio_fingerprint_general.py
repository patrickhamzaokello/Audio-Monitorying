import numpy as np
import librosa
import hashlib
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import argparse
from collections import defaultdict
import pickle

# Fixed optimal parameters based on industry standards
OPTIMAL_PARAMS = {
    'threshold': 0.35,        # Fixed threshold for peak detection
    'neighborhood_size': 10,  # Fixed neighborhood size for peak detection
    'fan_out': 15,           # Fixed fan-out for hash generation
    'time_window': 200       # Fixed time window for hash generation
}

def preprocess_audio(audio_path, sample_rate=22050):
    """
    Load and normalize audio file, convert to mono if needed
    """
    print(f"Loading audio file: {audio_path}")
    
    try:
        # Load audio file and resample
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        
        # Print some audio statistics
        print(f"Audio loaded: {len(audio)} samples at {sr}Hz")
        print(f"Duration: {len(audio)/sr:.2f} seconds")
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        return audio, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        raise

def create_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Create spectrogram using Short-Time Fourier Transform with optimized parameters
    """
    print("Creating spectrogram...")
    
    # Generate spectrogram with optimized parameters
    S = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to dB scale
    S_db = librosa.power_to_db(S, ref=np.max)
    
    print(f"Spectrogram shape: {S_db.shape}")
    
    return S_db

def find_peaks(spectrogram):
    """
    Find local peaks in the spectrogram using uniform parameters
    """
    # Use fixed parameters from OPTIMAL_PARAMS
    neighborhood_size = OPTIMAL_PARAMS['neighborhood_size']
    threshold = OPTIMAL_PARAMS['threshold']
    
    print(f"Finding peaks with fixed parameters: neighborhood_size={neighborhood_size}, threshold={threshold}")
    
    # Find local maximum points
    data_max = ndimage.maximum_filter(spectrogram, size=neighborhood_size)
    
    # For dB-scale: calculate threshold based on the range
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    db_range = max_val - min_val
    
    # Calculate threshold as a percentage of range above the minimum
    peak_threshold = min_val + (db_range * (1 - threshold))
    
    # Boolean array of peaks that exceed threshold
    maxima = (spectrogram == data_max) & (spectrogram > peak_threshold)
    
    # Get peak positions
    peak_positions = np.where(maxima)
    
    # Convert to list of (freq_bin, time_bin) tuples
    peaks = list(zip(peak_positions[0], peak_positions[1]))
    
    print(f"Found {len(peaks)} peaks")
    
    # Apply density-based filtering to ensure consistent peak density
    peaks = filter_peak_density(peaks, spectrogram.shape)
    
    return peaks

def filter_peak_density(peaks, spectrogram_shape):
    """
    Ensures consistent peak density across different audio samples
    """
    if not peaks:
        return []
    
    # Target peak density (per unit area in the spectrogram)
    target_density = 0.001  # peaks per pixel, adjust based on testing
    
    # Calculate current density
    spectrogram_area = spectrogram_shape[0] * spectrogram_shape[1]
    current_density = len(peaks) / spectrogram_area
    
    # If density is too high, sample peaks to reduce density
    if current_density > target_density:
        # Calculate how many peaks to keep
        keep_count = int(spectrogram_area * target_density)
        # Randomly sample peaks (could be improved with more sophisticated methods)
        keep_indices = np.random.choice(len(peaks), keep_count, replace=False)
        peaks = [peaks[i] for i in keep_indices]
        print(f"Filtered peaks to {len(peaks)} to maintain consistent density")
    
    return peaks

def generate_hashes(peaks):
    """
    Create fingerprint hashes from peaks using optimized, uniform parameters
    """
    # Use fixed parameters from OPTIMAL_PARAMS
    fan_out = OPTIMAL_PARAMS['fan_out']
    time_window = OPTIMAL_PARAMS['time_window']
    
    print(f"Generating hashes with fixed parameters: fan_out={fan_out}, time_window={time_window}")
    
    if not peaks:
        print("No peaks to generate hashes from!")
        return []
    
    # Create fingerprint hashes with optimized algorithm
    hashes = []
    
    # Sort peaks by time for more efficient processing
    peaks.sort(key=lambda p: p[1])
    
    # Optimized hash generation algorithm - similar to Shazam approach
    for i, anchor in enumerate(peaks):
        if i % 500 == 0 and i > 0:
            print(f"Processed {i}/{len(peaks)} anchor points")
        
        freq_anchor, time_anchor = anchor
        
        # Find target zone (peaks within time window after anchor)
        # Use binary search to find starting point more efficiently
        start_idx = i + 1
        while start_idx < len(peaks) and peaks[start_idx][1] - time_anchor < 1:  # Skip very close peaks
            start_idx += 1
        
        # Collect points within time window with adaptive fan out
        j = start_idx
        targets_added = 0
        
        # Enhanced: Distribute target points across the time window more evenly
        # Divide the time window into zones to ensure better distribution
        zones = 3  # Number of zones in the time window
        zone_targets = [0] * zones  # Count of targets in each zone
        max_per_zone = int(fan_out / zones) + 1  # Max targets per zone
        
        while j < len(peaks) and peaks[j][1] - time_anchor < time_window:
            # Determine which zone this target falls into
            delta_time = peaks[j][1] - time_anchor
            zone = min(int(delta_time * zones / time_window), zones - 1)
            
            # Only add if we haven't filled this zone yet
            if zone_targets[zone] < max_per_zone:
                freq_target, time_target = peaks[j]
                
                # Enhanced hashing algorithm: include more frequency information 
                # and use a more robust hash combination
                
                # Get frequency band information (divide frequency range into bands)
                freq_band_anchor = freq_anchor // 5  # Simple frequency banding
                freq_band_target = freq_target // 5
                
                # Create hash with more robust information
                hash_input = f"{freq_band_anchor}|{freq_band_target}|{delta_time}|{freq_anchor % 5}|{freq_target % 5}"
                
                # Create SHA-1 hash and take first 20 chars for efficiency
                hash_output = hashlib.sha1(hash_input.encode()).hexdigest()[:20]
                
                # Store hash with its time offset
                hashes.append((hash_output, time_anchor))
                
                # Update counters
                zone_targets[zone] += 1
                targets_added += 1
            
            j += 1
            
            # Stop if we've collected enough points across all zones
            if targets_added >= fan_out:
                break
    
    print(f"Generated {len(hashes)} hashes from {len(peaks)} peaks")
    return hashes

def visualize_spectrogram_and_peaks(spectrogram, sr, hop_length, peaks=None):
    """
    Visualize the spectrogram and optionally the detected peaks
    """
    plt.figure(figsize=(12, 8))
    
    # Plot spectrogram
    librosa.display.specshow(
        spectrogram, 
        sr=sr, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    
    # Optionally plot peaks
    if peaks and len(peaks) > 0:
        # Get peak coordinates
        peak_freqs = [p[0] for p in peaks]
        peak_times = [p[1] for p in peaks]
        
        # Plot peak points
        plt.scatter(
            peak_times, 
            peak_freqs, 
            color='red', 
            s=10, 
            alpha=0.5
        )
        plt.title(f'Spectrogram with {len(peaks)} peaks')
    
    plt.tight_layout()
    plt.savefig('spectrogram_with_peaks.png')
    print("Saved visualization to 'spectrogram_with_peaks.png'")
    plt.close()

def get_stable_song_id(file_path: str) -> int:
    """Generate consistent hash for the same file path"""
    return int(hashlib.sha256(file_path.encode()).hexdigest()[:15], 16)

def store_fingerprints_binary(audio_path, hashes, output_dir='fingerprints'):
    """Store fingerprints in binary file using pickle"""
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = os.path.basename(audio_path)
    output_path = os.path.join(output_dir, f"{file_name}.fp")
    
    data = {
        'file': audio_path,
        'duration': len(hashes) / (6000/180) if hashes else 0,
        'hashes': hashes
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Stored {len(hashes)} hashes in {output_path}")

def fingerprint_audio(audio_path, visualize=False):
    """
    Complete fingerprinting pipeline for audio file using fixed optimal parameters
    """
    print(f"\n=== Fingerprinting: {audio_path} with uniform parameters ===")
    
    # Process audio
    audio, sr = preprocess_audio(audio_path)
    
    # Create spectrogram
    hop_length = 512
    spec = create_spectrogram(audio, sr, hop_length=hop_length)
    
    # Find peaks with fixed parameters
    peaks = find_peaks(spec)
    
    # Generate hashes with fixed parameters
    hashes = generate_hashes(peaks)
    
    # Print statistics
    audio_length = len(audio) / sr  # in seconds
    hashes_per_sec = len(hashes) / audio_length if audio_length > 0 else 0
    
    print("\n=== Fingerprinting Results ===")
    print(f"Audio length: {audio_length:.2f} seconds")
    print(f"Generated {len(hashes)} hashes ({hashes_per_sec:.2f}/second)")
    
    # Visualize if requested
    if visualize and len(peaks) > 0:
        visualize_spectrogram_and_peaks(spec, sr, hop_length, peaks[:1000])  # Limit to 1000 peaks for clarity

    store_fingerprints_binary(audio_path, hashes)
    
    return hashes

def process_directory(directory_path, visualize=False):
    """
    Process all audio files in a directory using uniform parameters
    """
    print(f"\n=== Processing Directory: {directory_path} with uniform parameters ===")
    
    # Get list of audio files
    audio_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each file with the same fixed parameters
    results = {}
    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file}")
        try:
            hashes = fingerprint_audio(audio_file, visualize=visualize)
            results[audio_file] = len(hashes)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio fingerprinting tool using uniform parameters")
    parser.add_argument("input", help="Audio file or directory to process")
    parser.add_argument("--visualize", action="store_true", help="Visualize spectrogram and peaks")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        results = process_directory(args.input, visualize=args.visualize)
        print("\n=== Summary ===")
        for file, hash_count in results.items():
            print(f"{os.path.basename(file)}: {hash_count} hashes")
    else:
        fingerprint_audio(args.input, visualize=args.visualize)