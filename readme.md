# Audio Fingerprinting System Documentation

## Overview

This system provides robust audio fingerprinting and matching capabilities, similar to services like Shazam. It consists of two main components:

1. **Fingerprint Generator**: Processes audio files to create unique fingerprints stored in a database
2. **Matching API**: Allows clients to upload audio samples and find matches against the fingerprint database

## Table of Contents

- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Fingerprinting Audio Files](#fingerprinting-audio-files)
  - [Using the Matching API](#using-the-matching-api)
- [API Reference](#api-reference)
  - [Endpoints](#endpoints)
  - [Data Models](#data-models)
- [Technical Details](#technical-details)
  - [Fingerprinting Algorithm](#fingerprinting-algorithm)
  - [Matching Process](#matching-process)
  - [Optimal Parameters](#optimal-parameters)
- [Performance Considerations](#performance-considerations)
  - [Minimum Query Length](#minimum-query-length)
  - [Accuracy vs. Speed](#accuracy-vs-speed)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## System Architecture

```
┌───────────────────┐      ┌───────────────────┐
│                   │      │                   │
│ Audio Collection  │──┬──▶│ Fingerprint       │
│                   │  │   │ Generator         │
└───────────────────┘  │   │                   │
                       │   └─────────┬─────────┘
                       │             │
                       │             ▼
┌───────────────────┐  │   ┌───────────────────┐
│                   │  │   │                   │
│ Query Audio       │──┴──▶│ Fingerprint DB    │
│                   │      │                   │
└───────────────────┘      └─────────┬─────────┘
                                     │
                                     ▼
┌───────────────────┐      ┌───────────────────┐
│                   │      │                   │
│ Matching API      │◀─────│ Results           │
│                   │      │                   │
└───────────────────┘      └───────────────────┘
```

## Installation

### Prerequisites

- Python 3.7+
- Libraries:
  - numpy
  - librosa
  - scipy
  - fastapi
  - uvicorn
  - matplotlib (for visualization only)

### Setup

1. Install required packages:

```bash
pip install numpy librosa scipy fastapi uvicorn matplotlib pydantic
```

2. Clone or download the repository:

```bash
git clone https://your-repository-url.git
cd audio-fingerprinting
```

## Usage

### Fingerprinting Audio Files

To fingerprint a single audio file:

```bash
python audio_fingerprint_general.py path/to/audio.mp3
```

To fingerprint all audio files in a directory:

```bash
python audio_fingerprint_general.py path/to/directory --visualize
```

The `--visualize` flag will generate spectrogram visualizations with detected peaks.

### Using the Matching API

1. Start the API server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

2. Access the API documentation at `http://localhost:8000/docs`

3. Use the `/match` endpoint to upload audio and find matches

## API Reference

### Endpoints

#### POST `/match`

Matches an uploaded audio file against the fingerprint database.

**Parameters:**
- `file`: Audio file (MP3, WAV, FLAC, OGG)
- `min_matches`: Minimum number of matching hashes required (default: 5)

**Response:**
```json
{
  "query": "sample.mp3",
  "query_hashes": 305,
  "matches_found": 1,
  "processing_time_sec": 1.24,
  "matches": [
    {
      "song_id": "1a2b3c4d5e6f7g8h9i0j",
      "song_path": "/path/to/original.mp3",
      "confidence": 159,
      "offset": 1344,
      "match_count": 180,
      "temporal_info": {
        "match_start_seconds": 31.2,
        "match_duration_seconds": 3.7,
        "percentage_matched": 0.885,
        "hop_length": 512,
        "sample_rate": 22050
      }
    }
  ]
}
```

#### GET `/songs`

Lists all songs in the fingerprint database.

**Response:**
```json
{
  "count": 100,
  "songs": [
    {
      "song_id": "1a2b3c4d5e6f7g8h9i0j",
      "path": "/path/to/song.mp3",
      "fingerprint_count": 3540
    }
  ]
}
```

### Data Models

#### MatchResult

| Field | Type | Description |
|-------|------|-------------|
| song_id | string | Unique identifier for the matched song |
| song_path | string | File path to the matched audio file |
| confidence | integer | Number of consistent matching fingerprints |
| offset | integer | Frame offset between query and reference |
| match_count | integer | Total number of matching fingerprints |
| temporal_info | object | Temporal information about the match |

#### MatchResponse

| Field | Type | Description |
|-------|------|-------------|
| query | string | Name of the query file |
| query_hashes | integer | Number of fingerprints generated from query |
| matches_found | integer | Number of matches found |
| processing_time_sec | float | Processing time in seconds |
| matches | array | List of MatchResult objects |

## Technical Details

### Fingerprinting Algorithm

The system uses a constellation-based fingerprinting approach:

1. **Audio Preprocessing**:
   - Convert to mono
   - Normalize amplitude
   - Sample rate: 22050 Hz

2. **Spectrogram Generation**:
   - Mel-frequency spectrogram
   - n_fft: 2048
   - hop_length: 512
   - n_mels: 128

3. **Peak Detection**:
   - Local maximum filter with neighborhood size: 10
   - Adaptive threshold: 0.35 (relative to dB range)
   - Peak density control: 0.001 peaks per spectrogram pixel

4. **Hash Generation**:
   - For each anchor point, collect target points within time window (200 frames)
   - Use fan-out parameter (15) distributed across 3 zones
   - Generate hash combining frequency and time information
   - Hash format: `SHA1(freq_band_anchor|freq_band_target|delta_time|freq_anchor%5|freq_target%5)`

### Matching Process

1. Query audio is processed with identical parameters to the fingerprinting step
2. Hashes are matched against the database
3. Time offsets between matching points are calculated
4. Consistent time offsets indicate matches
5. Results are ranked by confidence (number of consistent matches)

### Optimal Parameters

The system uses fixed optimal parameters based on industry standards:

```python
OPTIMAL_PARAMS = {
    'threshold': 0.35,        # Threshold for peak detection
    'neighborhood_size': 10,  # Neighborhood size for peak detection
    'fan_out': 15,           # Fan-out for hash generation
    'time_window': 200       # Time window for hash generation
}
```

## Performance Considerations

### Minimum Query Length

For reliable matching:
- **Minimum**: 3-5 seconds
- **Optimal**: 8-10 seconds for high-quality recordings
- **Recommended for noisy environments**: 10-15 seconds

### Accuracy vs. Speed

Factors affecting performance:
- **Peak Density**: Higher density improves accuracy but increases processing time
- **Fan-out**: Higher values improve matching but increase database size
- **Time Window**: Larger windows improve robustness but increase computational cost

## Troubleshooting

### Common Issues

1. **No matches found**:
   - Ensure query length is sufficient (at least 5 seconds)
   - Check audio quality (high background noise reduces accuracy)
   - Verify that the reference track exists in the database

2. **Low confidence scores**:
   - Try a longer query sample
   - Use a sample from a distinctive part of the track (chorus rather than quiet intro)
   - Lower the `min_matches` parameter (caution: may increase false positives)

3. **Slow processing**:
   - Check database size
   - Consider optimizing database queries
   - Verify system hardware meets requirements

## Advanced Configuration

### Custom Parameters

While the system uses optimized fixed parameters, advanced users can modify:

- **Peak Detection**: Adjust `threshold` and `neighborhood_size` for different sensitivity
- **Hash Generation**: Modify `fan_out` and `time_window` to balance accuracy vs. resource usage
- **Matching**: Tune `min_matches` based on your application's needs

### Performance Optimization

For large-scale deployments:
- Consider database sharding
- Implement caching for frequent queries
- Use in-memory database for faster lookups
- Use parallel processing for fingerprint generation