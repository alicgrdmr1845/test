"""
Intake Watcher - Monitors filesystem for task submissions
Alternative to HTTP upload for environments with shared filesystems
"""

import os
import time
import json
import shutil
import uuid
import logging
from pathlib import Path
from typing import Dict, Any

import yaml
import redis
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
INTAKE_PATH = Path(os.getenv("INTAKE_PATH", "/app/intake"))
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "/app/storage"))

# Ensure directories exist
INTAKE_PATH.mkdir(parents=True, exist_ok=True)
STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# Redis connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


class TaskIntakeHandler(FileSystemEventHandler):
    """Handler for file system events in intake directory"""
    
    def on_created(self, event):
        """Process new files in intake directory"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process task.py files
        if file_path.name == "task.py":
            logger.info(f"New task detected: {file_path}")
            self.process_task_submission(file_path)
    
    def process_task_submission(self, task_path: Path):
        """Process a task submission from filesystem"""
        try:
            # Generate run ID
            run_id = f"run_{uuid.uuid4().hex[:8]}"
            
            # Look for accompanying run.yaml
            config_path = task_path.parent / "run.yaml"
            config_data = {}
            
            if config_path.exists():
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
            
            # Set defaults
            config_data.setdefault("run_name", run_id)
            config_data.setdefault("num_rounds", 1)
            config_data.setdefault("num_clients", 1)
            config_data.setdefault("framework", "pytorch")
            
            # Copy files to storage
            task_dir = STORAGE_PATH / "tasks" / run_id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(task_path, task_dir / "task.py")
            if config_path.exists():
                shutil.copy2(config_path, task_dir / "run.yaml")
            
            # Create run metadata
            run_info = {
                "run_id": run_id,
                "status": "queued",
                "task_path": str(task_dir / "task.py"),
                "config": json.dumps(config_data),
                "expected_clients": config_data["num_clients"],
                "received_updates": 0,
                "created_at": time.time(),
                "source": "filesystem"
            }
            
            # Store in Redis
            redis_client.hset(f"run:{run_id}", mapping=run_info)
            
            # Queue for processing
            task_queue_item = {
                "run_id": run_id,
                "round": 1,
                "config": config_data
            }
            redis_client.rpush("task_queue", json.dumps(task_queue_item))
            
            logger.info(f"Task queued successfully: {run_id}")
            
            # Move processed files to archive
            archive_dir = INTAKE_PATH / "processed" / run_id
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(task_path), str(archive_dir / "task.py"))
            if config_path.exists():
                shutil.move(str(config_path), str(archive_dir / "run.yaml"))
            
        except Exception as e:
            logger.error(f"Failed to process task submission: {e}")
            
            # Move to error directory
            error_dir = INTAKE_PATH / "errors"
            error_dir.mkdir(exist_ok=True)
            error_file = error_dir / f"{int(time.time())}_{task_path.name}"
            shutil.move(str(task_path), str(error_file))


def main():
    """Main watcher loop"""
    logger.info("Intake Watcher started")
    logger.info(f"Monitoring: {INTAKE_PATH}")
    logger.info(f"Storage: {STORAGE_PATH}")
    
    # Create subdirectories
    (INTAKE_PATH / "processed").mkdir(exist_ok=True)
    (INTAKE_PATH / "errors").mkdir(exist_ok=True)
    
    # Setup file system observer
    event_handler = TaskIntakeHandler()
    observer = Observer()
    observer.schedule(event_handler, str(INTAKE_PATH), recursive=False)
    
    # Start monitoring
    observer.start()
    logger.info("Watching for new tasks...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down intake watcher...")
        observer.stop()
    
    observer.join()


if __name__ == "__main__":
    main()