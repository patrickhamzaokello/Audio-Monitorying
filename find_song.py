from fastapi import FastAPI, UploadFile, File,Header, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import hashlib
from scipy import ndimage
import os
import pickle
from fastapi import Form
from collections import defaultdict
from typing import Dict, List, Optional
import tempfile
import time
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Audio Fingerprint Matching API",
    description="API for matching audio fingerprints against a pre-computed database",
    version="1.0.0"
)


# 👇 Configure CORS to allow requests from Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js frontend URL
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],  # Explicitly allow POST & OPTIONS
    allow_headers=["*"],  # Allow all headers (including X-Station-ID)
)

# Global variable to store loaded fingerprints
fingerprint_db: Dict[str, Dict] = {}
song_index: Dict[int, str] = {}


def get_song_id(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_fingerprint_db(pickle_dir: str = "fingerprints"):
    """Load all fingerprint pickle files into memory"""
    global fingerprint_db, song_index
    
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
        return
    
    for file in os.listdir(pickle_dir):
        if file.endswith(".fp"):
            try:
                with open(os.path.join(pickle_dir, file), 'rb') as f:
                    data = pickle.load(f)
                    file_hash = get_song_id(data['file'])  # Create unique hash for the file
                    fingerprint_db[file_hash] = data
                    song_index[file_hash] = data['file']
                    print(f"Loaded fingerprints for {data['file']}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    print(f"Loaded {len(fingerprint_db)} fingerprint files")


def generate_hashes_for_query(audio_path: str, sample_rate: int = 22050) -> List[tuple]:
    """Generate hashes for a query audio file using the same algorithm as the fingerprinter"""
    try:
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        audio = librosa.util.normalize(audio)
        
        # Create spectrogram
        hop_length = 512
        spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=128
        )
        S_db = librosa.power_to_db(spec, ref=np.max)
        
        # Find peaks using SAME parameters as the fingerprinter
        neighborhood_size = 10  # From OPTIMAL_PARAMS
        threshold = 0.35        # From OPTIMAL_PARAMS
        
        # Find local maximum points
        data_max = ndimage.maximum_filter(S_db, size=neighborhood_size)
        
        # For dB-scale: calculate threshold based on the range
        min_val = np.min(S_db)
        max_val = np.max(S_db)
        db_range = max_val - min_val
        
        # Calculate threshold as a percentage of range above the minimum
        peak_threshold = min_val + (db_range * (1 - threshold))
        
        # Boolean array of peaks that exceed threshold
        maxima = (S_db == data_max) & (S_db > peak_threshold)
        
        # Get peak positions
        peak_positions = np.where(maxima)
        
        # Convert to list of (freq_bin, time_bin) tuples
        peaks = list(zip(peak_positions[0], peak_positions[1]))
        
        # Apply density-based filtering to ensure consistent peak density
        if peaks:
            # Target peak density (per unit area in spectrogram)
            target_density = 0.001  # peaks per pixel
            
            # Calculate current density
            spectrogram_area = S_db.shape[0] * S_db.shape[1]
            current_density = len(peaks) / spectrogram_area
            
            # If density is too high, sample peaks to reduce density
            if current_density > target_density:
                # Calculate how many peaks to keep
                keep_count = int(spectrogram_area * target_density)
                # Randomly sample peaks
                keep_indices = np.random.choice(len(peaks), keep_count, replace=False)
                peaks = [peaks[i] for i in keep_indices]
        
        # Generate hashes with SAME parameters as fingerprinter
        fan_out = 15       # From OPTIMAL_PARAMS
        time_window = 200  # From OPTIMAL_PARAMS
        
        hashes = []
        
        # Sort peaks by time for more efficient processing
        peaks.sort(key=lambda p: p[1])
        
        # Optimized hash generation algorithm - same as fingerprinter
        for i, anchor in enumerate(peaks):
            freq_anchor, time_anchor = anchor
            
            # Find target zone (peaks within time window after anchor)
            start_idx = i + 1
            while start_idx < len(peaks) and peaks[start_idx][1] - time_anchor < 1:  # Skip very close peaks
                start_idx += 1
            
            # Collect points within time window with adaptive fan out
            j = start_idx
            targets_added = 0
            
            # Enhanced: Distribute target points across the time window more evenly
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
        
        return hashes
    except Exception as e:
        raise RuntimeError(f"Error processing query audio: {e}")

def add_temporal_info(match_result, query_hashes, reference_hashes, hop_length=512, sample_rate=22050):
    """
    Enhance match results with temporal information
    
    Args:
        match_result: Existing match dictionary
        query_hashes: List of (hash, offset) from query audio
        reference_hashes: List of (hash, offset) from matched song
        hop_length: STFT hop length (default 512)
        sample_rate: Audio sample rate (default 22050)
    """
    # Convert offset from frames to seconds
    offset_frames = match_result['offset']
    match_start_seconds = (offset_frames * hop_length) / sample_rate
    
    # Calculate match duration (using confidence as hash count)
    match_duration_seconds = (match_result['confidence'] * hop_length) / sample_rate
    
    # Calculate percentage matched in the reference segment
    ref_segment_hashes = [
        h for h in reference_hashes 
        if offset_frames <= h[1] <= offset_frames + match_result['confidence']
    ]
    percentage_matched = match_result['confidence'] / max(1, len(ref_segment_hashes))
    
    # Add to results
    match_result['temporal_info'] = {
        'match_start_seconds': round(match_start_seconds, 2),
        'match_duration_seconds': round(match_duration_seconds, 2),
        'percentage_matched': round(percentage_matched, 3),
        'hop_length': hop_length,  # For reference
        'sample_rate': sample_rate  # For reference
    }
    return match_result

def find_matches(query_hashes: List[tuple], min_matches: int = 5) -> List[dict]:
    """Find matches for query hashes in the fingerprint database"""
    matches = defaultdict(list)
    
    # Compare against all fingerprints in database
    for db_hash, db_data in fingerprint_db.items():
        # Create a hash lookup for faster matching
        db_hash_dict = defaultdict(list)
        for h, offset in db_data['hashes']:
            db_hash_dict[h].append(offset)
        
        # Match query hashes against database
        for q_hash, q_offset in query_hashes:
            if q_hash in db_hash_dict:
                for db_offset in db_hash_dict[q_hash]:
                    time_diff = db_offset - q_offset
                    matches[db_hash].append(time_diff)
    
    # Process matches to find songs with consistent time differences
    results = []
    for db_hash, diffs in matches.items():
        if len(diffs) >= min_matches:
            # Find the most common time difference
            diff_counts = defaultdict(int)
            for diff in diffs:
                diff_counts[diff] += 1
            
            # Get the top match
            consistent_matches = max(diff_counts.items(), key=lambda x: x[1])
            mode_diff, match_count = consistent_matches
            
            # Only include if we have enough consistent matches
            if match_count >= min_matches:
                results.append({
                    "song_id": db_hash,
                    "song_path": song_index.get(db_hash, "Unknown"),
                    "confidence": match_count,
                    "offset": int(mode_diff),
                    "match_count": len(diffs)  # Total matches, not just consistent ones
                })
    
    return sorted(results, key=lambda x: x['confidence'], reverse=True)

@app.on_event("startup")
async def startup_event():
    """Load fingerprint database on startup"""
    load_fingerprint_db()
    print("API ready with fingerprint database loaded")


# Define Pydantic models for response validation
class MatchResult(BaseModel):
    song_id: str
    song_path: str
    confidence: int
    offset: int
    match_count: int
    temporal_info: Optional[dict] = None  # Add this field

class MatchResponse(BaseModel):
    query: str
    query_hashes: int
    matches_found: int
    processing_time_sec: float
    matches: List[MatchResult]




# Assuming you have this model defined somewhere
class SampleCollectorResponse(BaseModel):
    message: str
    status: int

@app.post("/audio-samples", response_model=SampleCollectorResponse)
async def collect_audio_samples(
    file: UploadFile = File(...),
    stationId: str = Form(...), 
):
    # Create samples directory if it doesn't exist
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    # Validate station ID
    if not stationId:
        raise HTTPException(status_code=400, detail="Station ID header is required")
    
    # Generate a unique filename with timestamp and station ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{samples_dir}/{stationId}_{timestamp}.mp3"
    
    try:
        # Save the file
        with open(filename, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "audio file saved",
                "file_path": filename,
                "station_id": stationId,
                "size_bytes": len(content)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error saving audio file: {str(e)}"
        )

    

# Update the endpoint decorator
@app.post("/match", response_model=MatchResponse)
async def match_audio(file: UploadFile = File(...), min_matches: Optional[int] = 5):
    """
    Match uploaded audio against fingerprint database
    
    Parameters:
    - file: Audio file to match (mp3, wav, etc.)
    - min_matches: Minimum number of matching hashes to consider a match (default: 5)
    
    Returns:
    - Dictionary containing:
        - query: Original filename
        - query_hashes: Number of hashes generated
        - matches_found: Number of matches found
        - processing_time_sec: Time taken for processing
        - matches: List of match results

        Example:
        "confidence": 159 means 159 fingerprint points matched with consistent timing
        Example:
        "offset": 1344 →
        (1344 × 512)/22050 ≈ 31.2 seconds
        This means your query matches the reference track starting 31 seconds into the song
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # Generate hashes for query
        query_hashes = generate_hashes_for_query(tmp_path)
        os.unlink(tmp_path)  # Clean up temp file
        
        if not query_hashes:
            return JSONResponse(content={"message": "No fingerprints generated from query"}, status_code=400)
        
        # Find matches
        matches = find_matches(query_hashes, min_matches)

        enhanced_matches = []
        for match in matches:
            # Get reference hashes for the matched song
            ref_hashes = fingerprint_db[match['song_id']]['hashes']
            enhanced_match = add_temporal_info(match, query_hashes, ref_hashes)
            enhanced_matches.append(enhanced_match)

        # Prepare response
        processing_time = time.time() - start_time
        response = MatchResponse(
            query=file.filename,
            query_hashes=len(query_hashes),
            matches_found=len(enhanced_matches),
            processing_time_sec=round(processing_time, 2),
            matches=[
                MatchResult(
                    song_id=match["song_id"],
                    song_path=match["song_path"],
                    confidence=match["confidence"],
                    offset=match["offset"],
                    match_count=match["match_count"],
                    temporal_info=match.get("temporal_info")
                ) for match in enhanced_matches
            ]
        )

        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SongInfo(BaseModel):
    song_id: str
    path: str
    fingerprint_count: int

class SongListResponse(BaseModel):
    count: int
    songs: List[SongInfo]

@app.get("/songs", response_model=SongListResponse)
async def list_songs():
    """List all songs in the fingerprint database"""
    songs = []
    for song_hash, song_path in song_index.items():
        songs.append(SongInfo(
            song_id=str(song_hash),
            path=song_path,
            fingerprint_count=len(fingerprint_db.get(song_hash, {}).get('hashes', []))
        ))
    return SongListResponse(count=len(songs), songs=songs)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)