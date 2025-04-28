#!/usr/bin/env python3
"""
Radio Airplay Tracking - Sample Processor

This script processes audio samples collected from radio stations, 
matches them against the fingerprint database, and tracks song airplay
while avoiding duplicate counts within a specified time window.
"""

import os
import json
import time
import logging
import requests
import datetime
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("airplay_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("airplay_processor")

# Default configuration
DEFAULT_CONFIG = {
    "api_url": "http://localhost:8000",  # API base URL
    "samples_dir": "samples",            # Directory with audio samples
    "processed_db": "processed_samples.db",  # SQLite DB to track processed files
    "min_confidence": 10,                # Minimum confidence for a match
    "deduplication_window": 300,         # Time window in seconds (5 minutes)
    "batch_size": 20,                    # Number of samples to process in one batch
    "match_endpoint": "/match",          # Endpoint for matching audio
    "output_dir": "reports",             # Directory for airplay reports
    "delay_between_requests": 0.5        # Delay between API requests (seconds)
}

class AirplayProcessor:
    """Processes audio samples and tracks song airplay"""
    
    def __init__(self, config: Dict = None):
        """Initialize the processor with configuration"""
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Ensure directories exist
        os.makedirs(self.config["samples_dir"], exist_ok=True)
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized AirplayProcessor with config: {self.config}")
    
    def _init_database(self):
        """Initialize the SQLite database for tracking processed samples and airplay"""
        self.conn = sqlite3.connect(self.config["processed_db"])
        cursor = self.conn.cursor()
        
        # Create table for processed files
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            id INTEGER PRIMARY KEY,
            filename TEXT UNIQUE,
            processed_at TIMESTAMP,
            match_success BOOLEAN,
            song_id TEXT,
            song_path TEXT,
            confidence INTEGER
        )
        ''')
        
        # Create table for airplay tracking
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS airplay (
            id INTEGER PRIMARY KEY,
            station_id TEXT,
            song_id TEXT,
            song_path TEXT,
            detected_at TIMESTAMP,
            confidence INTEGER,
            UNIQUE(station_id, song_id, detected_at)
        )
        ''')
        
        self.conn.commit()
        logger.info("Database initialized")
    
    def _get_unprocessed_samples(self, limit: int = None) -> List[str]:
        """Get a list of unprocessed sample files"""
        cursor = self.conn.cursor()
        
        # Get all files in the samples directory
        sample_files = []
        for file in os.listdir(self.config["samples_dir"]):
            if file.endswith(".mp3") or file.endswith(".wav"):
                sample_files.append(file)
        
        # Filter out already processed files
        processed_files = []
        cursor.execute("SELECT filename FROM processed_files")
        for row in cursor.fetchall():
            processed_files.append(row[0])
        
        unprocessed = [f for f in sample_files if f not in processed_files]
        
        if limit:
            return unprocessed[:limit]
        return unprocessed
    
    def _extract_station_id(self, filename: str) -> str:
        """Extract the station ID from the filename
        Expected format: stationId_YYYYMMDD_HHMMSS.mp3
        """
        try:
            # Split by underscore and take the first part
            station_id = filename.split('_')[0]
            return station_id
        except Exception as e:
            logger.warning(f"Could not extract station ID from {filename}: {e}")
            return "unknown"
    
    def _extract_timestamp(self, filename: str) -> datetime.datetime:
        """Extract the timestamp from the filename
        Expected format: stationId_YYYYMMDD_HHMMSS.mp3
        """
        try:
            # Extract the date and time parts
            parts = filename.split('_')
            date_part = parts[1]
            time_part = parts[2].split('.')[0]  # Remove file extension
            
            # Parse into datetime
            dt_str = f"{date_part}_{time_part}"
            return datetime.datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
        except Exception as e:
            logger.warning(f"Could not extract timestamp from {filename}: {e}")
            return datetime.datetime.now()  # Fallback to current time
    
    def _is_duplicate(self, station_id: str, song_id: str, timestamp: datetime.datetime) -> bool:
        """Check if this song was already detected on this station within the window"""
        cursor = self.conn.cursor()
        
        # Calculate the time window
        window_start = timestamp - datetime.timedelta(seconds=self.config["deduplication_window"])
        
        cursor.execute("""
        SELECT COUNT(*) FROM airplay
        WHERE station_id = ? AND song_id = ? AND detected_at > ?
        """, (station_id, song_id, window_start))
        
        count = cursor.fetchone()[0]
        return count > 0
    
    def _match_audio_sample(self, file_path: str) -> Dict:
        """Send the audio sample to the match endpoint and get results"""
        try:
            url = f"{self.config['api_url']}{self.config['match_endpoint']}"
            params = {"min_matches": self.config["min_confidence"]}
            
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(url, files=files, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error matching {file_path}: {response.status_code} - {response.text}")
                return {"matches_found": 0, "matches": []}
        except Exception as e:
            logger.error(f"Exception during match for {file_path}: {str(e)}")
            return {"matches_found": 0, "matches": []}
    
    def _record_processed_file(self, filename: str, match_result: Dict = None):
        """Record that a file has been processed"""
        cursor = self.conn.cursor()
        
        # Default values for no match
        match_success = False
        song_id = None
        song_path = None
        confidence = 0
        
        # Update if we have a match
        if match_result and match_result.get("matches_found", 0) > 0:
            best_match = match_result["matches"][0]
            match_success = True
            song_id = best_match["song_id"]
            song_path = best_match["song_path"]
            confidence = best_match["confidence"]
        
        cursor.execute("""
        INSERT INTO processed_files 
        (filename, processed_at, match_success, song_id, song_path, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            filename, 
            datetime.datetime.now(), 
            match_success, 
            song_id, 
            song_path, 
            confidence
        ))
        
        self.conn.commit()
    
    def _record_airplay(self, station_id: str, song_id: str, song_path: str, 
                        timestamp: datetime.datetime, confidence: int):
        """Record a song airplay event"""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
            INSERT INTO airplay
            (station_id, song_id, song_path, detected_at, confidence)
            VALUES (?, ?, ?, ?, ?)
            """, (station_id, song_id, song_path, timestamp, confidence))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # This could happen with the UNIQUE constraint if it's a duplicate
            logger.debug(f"Duplicate airplay entry: {station_id} - {song_id} - {timestamp}")
            return False
    
    def process_batch(self, batch_size: int = None) -> int:
        """Process a batch of unprocessed samples
        
        Args:
            batch_size: Number of samples to process (default: from config)
            
        Returns:
            Number of samples processed
        """
        if batch_size is None:
            batch_size = self.config["batch_size"]
        
        unprocessed = self._get_unprocessed_samples(batch_size)
        logger.info(f"Found {len(unprocessed)} unprocessed samples, processing up to {batch_size}")
        
        processed_count = 0
        for filename in unprocessed:
            file_path = os.path.join(self.config["samples_dir"], filename)
            
            # Extract metadata from filename
            station_id = self._extract_station_id(filename)
            timestamp = self._extract_timestamp(filename)
            
            # Match the audio sample
            match_result = self._match_audio_sample(file_path)
            
            # Record that we processed this file
            self._record_processed_file(filename, match_result)
            processed_count += 1
            
            # If we have a match, record the airplay
            if match_result and match_result.get("matches_found", 0) > 0:
                best_match = match_result["matches"][0]
                song_id = best_match["song_id"]
                song_path = best_match["song_path"]
                confidence = best_match["confidence"]
                
                # Only record if confidence meets threshold and it's not a duplicate
                if confidence >= self.config["min_confidence"]:
                    if not self._is_duplicate(station_id, song_id, timestamp):
                        self._record_airplay(
                            station_id, song_id, song_path, timestamp, confidence
                        )
                        logger.info(f"Recorded airplay: {station_id} - {song_path}")
                    else:
                        logger.info(f"Skipped duplicate: {station_id} - {song_path}")
            
            # Add a small delay between requests to avoid overloading the API
            time.sleep(self.config["delay_between_requests"])
        
        return processed_count
    
    def generate_report(self, start_time: datetime.datetime = None, 
                        end_time: datetime.datetime = None, 
                        output_file: str = None) -> Dict:
        """Generate an airplay report for a time period
        
        Args:
            start_time: Start of the reporting period (default: last 24 hours)
            end_time: End of the reporting period (default: now)
            output_file: Path to save the report JSON (default: based on time period)
            
        Returns:
            Dict with the report data
        """
        if end_time is None:
            end_time = datetime.datetime.now()
        
        if start_time is None:
            start_time = end_time - datetime.timedelta(days=1)
        
        cursor = self.conn.cursor()
        
        # Get all airplay events in the time period
        cursor.execute("""
        SELECT station_id, song_id, song_path, detected_at, confidence
        FROM airplay
        WHERE detected_at BETWEEN ? AND ?
        ORDER BY detected_at
        """, (start_time, end_time))
        
        # Organize data by station and song
        stations = {}
        songs = {}
        all_plays = []
        
        for row in cursor.fetchall():
            station_id, song_id, song_path, detected_at, confidence = row
            
            # Parse datetime from string if needed
            if isinstance(detected_at, str):
                detected_at = datetime.datetime.fromisoformat(detected_at)
            
            # For the all plays list
            play_entry = {
                "station_id": station_id,
                "song_id": song_id,
                "song_path": song_path,
                "detected_at": detected_at.isoformat(),
                "confidence": confidence
            }
            all_plays.append(play_entry)
            
            # For stations summary
            if station_id not in stations:
                stations[station_id] = {"total_plays": 0, "songs": {}}
            
            stations[station_id]["total_plays"] += 1
            if song_id not in stations[station_id]["songs"]:
                stations[station_id]["songs"][song_id] = 0
            stations[station_id]["songs"][song_id] += 1
            
            # For songs summary
            if song_id not in songs:
                songs[song_id] = {
                    "song_path": song_path,
                    "total_plays": 0,
                    "stations": {}
                }
            
            songs[song_id]["total_plays"] += 1
            if station_id not in songs[song_id]["stations"]:
                songs[song_id]["stations"][station_id] = 0
            songs[song_id]["stations"][station_id] += 1
        
        # Sort songs by total plays
        top_songs = sorted(
            [{"song_id": id, **data} for id, data in songs.items()],
            key=lambda x: x["total_plays"],
            reverse=True
        )
        
        # Create the full report
        report = {
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": {
                "total_plays": len(all_plays),
                "unique_stations": len(stations),
                "unique_songs": len(songs)
            },
            "top_songs": top_songs[:20],  # Top 20 songs
            "station_summary": stations,
            "all_plays": all_plays
        }
        
        # Save to file if requested
        if output_file:
            output_path = output_file
        else:
            # Generate filename based on time period
            start_str = start_time.strftime("%Y%m%d")
            end_str = end_time.strftime("%Y%m%d")
            output_path = os.path.join(
                self.config["output_dir"], 
                f"airplay_report_{start_str}_to_{end_str}.json"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated report saved to {output_path}")
        
        return report
    
    def run_continuous(self, interval: int = 60):
        """Run the processor continuously
        
        Args:
            interval: Time between processing batches in seconds
        """
        logger.info(f"Starting continuous processing with interval {interval} seconds")
        
        try:
            while True:
                start_time = time.time()
                processed = self.process_batch()
                
                logger.info(f"Processed {processed} samples")
                
                # Generate a daily report at midnight
                now = datetime.datetime.now()
                if now.hour == 0 and now.minute < interval/60:
                    yesterday = now - datetime.timedelta(days=1)
                    yesterday_start = datetime.datetime(
                        yesterday.year, yesterday.month, yesterday.day, 0, 0, 0
                    )
                    yesterday_end = datetime.datetime(
                        now.year, now.month, now.day, 0, 0, 0
                    )
                    
                    self.generate_report(yesterday_start, yesterday_end)
                
                # Calculate sleep time to maintain the interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Processor stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous processing: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Process audio samples and track radio airplay")
    
    # Operational mode arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--process", action="store_true", help="Process unprocessed samples")
    mode_group.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    mode_group.add_argument("--report", action="store_true", help="Generate an airplay report")
    
    # Configuration
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--api-url", help="API base URL")
    parser.add_argument("--samples-dir", help="Directory with audio samples")
    parser.add_argument("--output-dir", help="Directory for output reports")
    
    # Processing options
    parser.add_argument("--batch-size", type=int, help="Number of samples to process in one batch")
    parser.add_argument("--interval", type=int, default=60, help="Interval between batches in continuous mode (seconds)")
    parser.add_argument("--min-confidence", type=int, help="Minimum confidence score for a match")
    parser.add_argument("--window", type=int, help="Deduplication window in seconds")
    
    # Report options
    parser.add_argument("--start-date", help="Start date for report (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for report (YYYY-MM-DD)")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Override with command-line arguments
    if args.api_url:
        config["api_url"] = args.api_url
    if args.samples_dir:
        config["samples_dir"] = args.samples_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.min_confidence:
        config["min_confidence"] = args.min_confidence
    if args.window:
        config["deduplication_window"] = args.window
    
    # Initialize the processor
    processor = AirplayProcessor(config)
    
    # Execute the selected mode
    if args.process:
        # Process a single batch
        processed = processor.process_batch()
        print(f"Processed {processed} samples")
    
    elif args.continuous:
        # Run in continuous mode
        processor.run_continuous(args.interval)
    
    elif args.report:
        # Generate a report
        start_date = None
        end_date = None
        
        if args.start_date:
            start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
        
        if args.end_date:
            end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
            # Set to end of day
            end_date = end_date.replace(hour=23, minute=59, second=59)
        
        report = processor.generate_report(start_date, end_date, args.output)
        print(f"Generated report with {report['summary']['total_plays']} plays")
    
    else:
        # Default action: print help
        parser.print_help()

if __name__ == "__main__":
    main()