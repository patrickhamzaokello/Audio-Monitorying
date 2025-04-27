import numpy as np
import librosa
import hashlib
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import argparse
from collections import defaultdict

def preprocess_audio(audio_path, sample_rate=22050):
    """
    Load and normalize audio file, convert to mono if needed
    """
    print(f"Loading audio file: {audio_path}")
    print(f"File exists: {os.path.exists(audio_path)}")
    
    try:
        # Load audio file and resample
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        
        # Print some audio statistics
        print(f"Audio loaded: {len(audio)} samples at {sr}Hz")
        print(f"Duration: {len(audio)/sr:.2f} seconds")
        print(f"Audio range: {audio.min():.4f} to {audio.max():.4f}")
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        return audio, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        raise

def create_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Create spectrogram using Short-Time Fourier Transform
    """
    print("Creating spectrogram...")
    
    # Generate spectrogram
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
    print(f"Spectrogram range: {S_db.min():.2f} to {S_db.max():.2f} dB")
    
    return S_db

def find_peaks(spectrogram, neighborhood_size=10, threshold=0.3):
    """
    Find local peaks in the spectrogram using an approach suitable for dB-scale data
    
    Args:
        spectrogram: dB-scale spectrogram (values typically between -80 and 0)
        neighborhood_size: Size of maximum filter neighborhood
        threshold: Value between 0 and 1, representing the percentage of the dB range
                  to use as threshold (e.g., 0.7 means top 30% of the range)
    """
    print(f"Finding peaks with neighborhood_size={neighborhood_size}, threshold={threshold}")
    
    # Find local maximum points
    data_max = ndimage.maximum_filter(spectrogram, size=neighborhood_size)
    
    # For dB-scale: calculate threshold based on the range
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    db_range = max_val - min_val  # This will be a positive number
    
    # Calculate threshold as a percentage of range above the minimum
    # threshold=0.3 means top 70% of the range, threshold=0.7 means top 30%
    peak_threshold = min_val + (db_range * (1 - threshold))
    
    print(f"dB range: {min_val:.2f} to {max_val:.2f} dB (range: {db_range:.2f} dB)")
    print(f"Peak threshold: {peak_threshold:.2f} dB (top {threshold*100:.0f}% of range)")
    
    # Boolean array of peaks that exceed threshold
    maxima = (spectrogram == data_max) & (spectrogram > peak_threshold)
    
    # Get peak positions
    peak_positions = np.where(maxima)
    
    # Convert to list of (freq_bin, time_bin) tuples
    peaks = list(zip(peak_positions[0], peak_positions[1]))
    
    print(f"Found {len(peaks)} peaks")
    
    # If no peaks found, try more aggressive parameters
    if len(peaks) == 0:
        print("No peaks found with current parameters. Try decreasing threshold or neighborhood_size.")
        
        # Optional: Automatically try with more relaxed parameters
        if threshold < 0.9:  # Only try if we haven't already used a very low threshold
            new_threshold = min(threshold + 0.2, 0.9)  # Increase threshold by 0.2 (lower selectivity)
            print(f"Automatically retrying with threshold={new_threshold}...")
            return find_peaks(spectrogram, neighborhood_size, new_threshold)
    
    return peaks

def generate_hashes(peaks, fan_out=15, time_window=200):
    """
    Create fingerprint hashes from peaks using a constellation approach
    """
    print(f"Generating hashes with fan_out={fan_out}, time_window={time_window}")
    
    if not peaks:
        print("No peaks to generate hashes from!")
        return []
    
    # Create fingerprint hashes
    hashes = []
    hash_count = 0
    
    # Sort peaks by time for more efficient processing
    peaks.sort(key=lambda p: p[1])
    
    for i, anchor in enumerate(peaks):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{len(peaks)} anchor points, generated {hash_count} hashes so far")
        
        freq_anchor, time_anchor = anchor
        
        # Find target zone (peaks within time window after anchor)
        # Optimized to avoid full scan of all peaks for each anchor
        j = i + 1
        target_points = []
        
        # Collect points within time window
        while j < len(peaks) and peaks[j][1] - time_anchor < time_window:
            target_points.append(peaks[j])
            j += 1
            
            # Stop if we've collected enough points
            if len(target_points) >= fan_out:
                break
        
        # Create hash for each anchor-point pair
        for target in target_points:
            freq_target, time_target = target
            
            # Calculate time delta
            delta_time = time_target - time_anchor
            
            # Create hash with frequency and time delta information
            hash_input = f"{freq_anchor}|{freq_target}|{delta_time}"
            
            # Create SHA-1 hash
            hash_output = hashlib.sha1(hash_input.encode()).hexdigest()
            
            # Store hash with its time offset
            hashes.append((hash_output, time_anchor))
            hash_count += 1
    
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

import pickle

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

def fingerprint_audio(audio_path, neighborhood_size=10, threshold=0.3, 
                     fan_out=15, time_window=200, visualize=False):
    """
    Complete fingerprinting pipeline for audio file
    """
    print(f"\n=== Fingerprinting: {audio_path} ===")
    
    # Process audio
    audio, sr = preprocess_audio(audio_path)
    
    # Create spectrogram
    hop_length = 512
    spec = create_spectrogram(audio, sr, hop_length=hop_length)
    
    # Find peaks
    peaks = find_peaks(spec, neighborhood_size, threshold)
    
    # Generate hashes
    hashes = generate_hashes(peaks, fan_out, time_window)
    
    # Print statistics
    audio_length = len(audio) / sr  # in seconds
    hashes_per_sec = len(hashes) / audio_length if audio_length > 0 else 0
    expected_hashes = audio_length * (6000 / 180)  # 6000 hashes per 3 minutes
    
    print("\n=== Fingerprinting Results ===")
    print(f"Audio length: {audio_length:.2f} seconds")
    print(f"Generated {len(hashes)} hashes ({hashes_per_sec:.2f}/second)")
    print(f"Expected ~{expected_hashes:.0f} hashes for target density")
    
    # Calculate hash density ratio compared to target
    target_density = 6000 / 180  # hashes per second
    actual_density = hashes_per_sec
    density_ratio = actual_density / target_density if target_density > 0 else 0
    
    print(f"Hash density ratio: {density_ratio:.2f}x target")
    
    # Visualize if requested
    if visualize and len(peaks) > 0:
        visualize_spectrogram_and_peaks(spec, sr, hop_length, peaks[:1000])  # Limit to 1000 peaks for clarity

    store_fingerprints_binary(audio_path, hashes)
    
    return hashes

def tune_parameters(audio_path, target_density=6000/180):
    """
    Tune parameters to achieve target hash density
    """
    print(f"\n=== Parameter Tuning: {audio_path} ===")
    print(f"Target density: {target_density:.2f} hashes/second")
    
    # Calculate target hash count
    audio, sr = preprocess_audio(audio_path)
    duration = len(audio) / sr
    target_hash_count = duration * target_density
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Target hash count: {target_hash_count:.0f}")
    
    # Create spectrogram once
    hop_length = 512
    spec = create_spectrogram(audio, sr, hop_length=hop_length)
    
    # Initial parameters
    best_params = {
        'threshold': 0.3,
        'neighborhood_size': 10,
        'fan_out': 15,
        'time_window': 200
    }
    
    # Parameter ranges to try
    thresholds = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
    neighborhood_sizes = [20, 15, 10, 5, 3]
    fan_outs = [10, 15, 20, 25, 30]
    time_windows = [100, 200, 300]
    
    best_score = float('inf')  # Initialize with worst possible score
    best_hash_count = 0
    
    # Try different parameter combinations
    print("\n=== Testing Parameter Combinations ===")
    for threshold in thresholds:
        for neighborhood_size in neighborhood_sizes:
            # Find peaks with current parameters
            peaks = find_peaks(spec, neighborhood_size, threshold)
            
            # Skip if no peaks found
            if not peaks:
                print(f"No peaks found with threshold={threshold}, neighborhood_size={neighborhood_size}")
                continue
            
            # Only test fan_out and time_window if we have peaks
            for fan_out in fan_outs:
                for time_window in time_windows:
                    # Generate hashes with current parameters
                    hashes = generate_hashes(peaks, fan_out, time_window)
                    hash_count = len(hashes)
                    
                    # Calculate density
                    density = hash_count / duration if duration > 0 else 0
                    
                    # Calculate how close we are to target density
                    density_score = abs(density - target_density) / target_density
                    
                    print(f"Parameters: threshold={threshold}, neighborhood_size={neighborhood_size}, "
                          f"fan_out={fan_out}, time_window={time_window}")
                    print(f"  Generated {hash_count} hashes ({density:.2f}/second), "
                          f"target score: {density_score:.2f}")
                    
                    # Update best parameters if this is better
                    if density > 0 and density_score < best_score:
                        best_score = density_score
                        best_hash_count = hash_count
                        best_params = {
                            'threshold': threshold,
                            'neighborhood_size': neighborhood_size,
                            'fan_out': fan_out,
                            'time_window': time_window
                        }
                    
                    # Stop early if we're within 10% of target
                    if density_score < 0.1:
                        print("  Found parameters within 10% of target density!")
                        break
                
                # Stop early if we're within 10% of target
                if best_score < 0.1:
                    break
            
            # Stop early if we're within 10% of target
            if best_score < 0.1:
                break
    
    print("\n=== Tuning Results ===")
    print(f"Best parameters: {best_params}")
    print(f"Generated {best_hash_count} hashes (target: {target_hash_count:.0f})")
    print(f"Best score: {best_score:.2f} (lower is better)")
    
    return best_params

def process_directory(directory_path, **kwargs):
    """
    Process all audio files in a directory
    """
    print(f"\n=== Processing Directory: {directory_path} ===")
    
    # Get list of audio files
    audio_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each file
    results = {}
    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file}")
        try:
            hashes = fingerprint_audio(audio_file, **kwargs)
            results[audio_file] = len(hashes)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    return results


params = tune_parameters("content/ONAWULIRA (ft.Aaronx) .mp3")
fingerprint_audio("content/ONAWULIRA (ft.Aaronx) .mp3", **params, visualize="visualize")