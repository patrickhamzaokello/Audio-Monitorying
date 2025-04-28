# Radio Airplay Tracking System

This system processes audio samples collected from radio stations, identifies songs using acoustic fingerprinting, and generates reports on radio airplay while avoiding duplicate counts.

## How It Works

1. Radio stations send 8-second audio samples to your backend API
2. The samples are stored in the `samples` directory
3. The sample processor script:
   - Processes unprocessed audio samples
   - Matches them against your fingerprint database
   - Tracks song airplay while avoiding duplicate counts
   - Generates reports on song plays across stations

## Features

- **Deduplication**: Counts each song only once per station within a configurable time window
- **Batch Processing**: Processes samples in batches to manage system resources
- **Continuous Mode**: Runs continuously to process new samples as they arrive
- **Reporting**: Generates comprehensive reports on airplay by song and station
- **Database Tracking**: Keeps track of processed files and detected airplay

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required dependencies:

```bash
pip install requests
```

## Usage

### Configuration

You can configure the processor using command-line arguments or a JSON configuration file:

```json
{
  "api_url": "http://localhost:8000",
  "samples_dir": "samples",
  "processed_db": "processed_samples.db",
  "min_confidence": 10,
  "deduplication_window": 300,
  "batch_size": 20,
  "output_dir": "reports"
}
```

### Command-Line Options

#### Process a batch of samples:

```bash
python airplay_processor.py --process --batch-size 50
```

#### Run in continuous mode:

```bash
python airplay_processor.py --continuous --interval 60
```

#### Generate a report:

```bash
python airplay_processor.py --report --start-date 2025-04-20 --end-date 2025-04-27
```

#### Use a custom configuration file:

```bash
python airplay_processor.py --continuous --config airplay_config.json
```

### Advanced Options

- `--api-url`: API base URL (default: http://localhost:8000)
- `--samples-dir`: Directory with audio samples (default: samples)
- `--output-dir`: Directory for output reports (default: reports)
- `--min-confidence`: Minimum confidence score for a match (default: 10)
- `--window`: Deduplication window in seconds (default: 300)
- `--interval`: Interval between batches in continuous mode (default: 60)

## Reports

The system generates comprehensive reports in JSON format with:

- Summary statistics (total plays, unique stations, unique songs)
- Top songs by play count
- Play counts by station
- Detailed list of all plays with timestamps

Example report structure:

```json
{
  "report_period": {
    "start": "2025-04-20T00:00:00",
    "end": "2025-04-27T23:59:59"
  },
  "summary": {
    "total_plays": 1250,
    "unique_stations": 15,
    "unique_songs": 320
  },
  "top_songs": [
    {
      "song_id": "a1b2c3d4...",
      "song_path": "path/to/song.mp3",
      "total_plays": 42,
      "stations": {
        "station1": 12,
        "station2": 10,
        "station3": 20
      }
    },
    ...
  ],
  "station_summary": {
    "station1": {
      "total_plays": 230,
      "songs": {
        "song_id1": 12,
        "song_id2": 8,
        ...
      }
    },
    ...
  },
  "all_plays": [
    {
      "station_id": "station1",
      "song_id": "song_id1",
      "song_path": "path/to/song.mp3",
      "detected_at": "2025-04-20T13:45:22",
      "confidence": 35
    },
    ...
  ]
}
```

## Integration

This script is designed to work with your existing audio fingerprinting API. It expects the following:

1. Samples in the format: `{stationId}_{timestamp}.mp3` in the samples directory
2. A `/match` endpoint on your API that accepts audio files and returns match results

## Customization

You can customize the script to fit your specific needs:

- Adjust the deduplication window to match your station's rotation patterns
- Configure the minimum confidence threshold based on your fingerprinting accuracy
- Modify the report format to include additional metrics