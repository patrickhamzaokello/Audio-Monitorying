from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import hashlib
from scipy import ndimage
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional
import tempfile
import time
from pydantic import BaseModel


app = FastAPI(
    title="Audio Fingerprint Matching API",
    description="API for matching audio fingerprints against a pre-computed database",
    version="1.0.0"
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
    """Generate hashes for a query audio file (same as your existing code)"""
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
        
        # Find peaks
        data_max = ndimage.maximum_filter(S_db, size=10)
        min_val = np.min(S_db)
        max_val = np.max(S_db)
        db_range = max_val - min_val
        peak_threshold = min_val + (db_range * 0.7)  # Using 0.3 threshold equivalent
        maxima = (S_db == data_max) & (S_db > peak_threshold)
        peak_positions = np.where(maxima)
        peaks = list(zip(peak_positions[0], peak_positions[1]))
        
        # Generate hashes
        hashes = []
        peaks.sort(key=lambda p: p[1])  # Sort by time
        
        for i, anchor in enumerate(peaks):
            freq_anchor, time_anchor = anchor
            
            # Find target points within time window
            j = i + 1
            while j < len(peaks) and (peaks[j][1] - time_anchor) < 200:
                freq_target, time_target = peaks[j]
                delta_time = time_target - time_anchor
                hash_input = f"{freq_anchor}|{freq_target}|{delta_time}"
                hash_output = hashlib.sha1(hash_input.encode()).hexdigest()
                hashes.append((hash_output, time_anchor))
                j += 1
        
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
    """Modified to work with string IDs"""
    matches = defaultdict(list)
    
    # Compare against all fingerprints in database
    for db_hash, db_data in fingerprint_db.items():  # db_hash is now a string
        db_hash_set = {h[0] for h in db_data['hashes']}
        
        for q_hash, q_offset in query_hashes:
            if q_hash in db_hash_set:
                for db_h, db_offset in db_data['hashes']:
                    if db_h == q_hash:
                        time_diff = db_offset - q_offset
                        matches[db_hash].append(time_diff)
    
    # Process matches (same as before but keeping string IDs)
    results = []
    for db_hash, diffs in matches.items():
        if len(diffs) >= min_matches:
            mode_diff = max(set(diffs), key=diffs.count)
            results.append({
                "song_id": db_hash,  # Keep as string
                "song_path": song_index.get(db_hash, "Unknown"),
                "confidence": len(diffs),
                "offset": int(mode_diff),
                "match_count": len(diffs)
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
This means your query matches the reference track starting 31 seconds into the son
    """
    start_time = time.time()
    
    # Validate file type
    # if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
    #     raise HTTPException(status_code=400, detail="Unsupported file format")
    
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
                    temporal_info=match.get("temporal_info")  # Add this line
                ) for match in enhanced_matches  # Use enhanced_matches here
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