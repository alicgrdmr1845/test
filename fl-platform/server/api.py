"""
Minimal FL Platform Server API
Handles client polling, task submission, and result retrieval
"""

import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import yaml
import redis
import numpy as np

app = FastAPI(title="FL Platform", version="1.0.0")

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
API_TOKEN = os.getenv("API_TOKEN", "demo-token-123")
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "/app/storage"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

# Ensure storage directories exist
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
(STORAGE_PATH / "tasks").mkdir(exist_ok=True)
(STORAGE_PATH / "updates").mkdir(exist_ok=True)
(STORAGE_PATH / "artifacts").mkdir(exist_ok=True)

# Redis connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# --- Models ---

class ClientRegistration(BaseModel):
    client_id: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = {}
    dataset_labels: Optional[List[str]] = []

class ClientHeartbeat(BaseModel):
    client_id: str
    status: Optional[str] = "healthy"

class UpdateUpload(BaseModel):
    metrics: Dict[str, float]
    num_examples: int

class RunStatus(BaseModel):
    run_id: str
    status: str
    expected_clients: int
    received_updates: int
    done: bool
    artifacts: Dict[str, str]
    metrics: Optional[Dict[str, Any]] = None

# --- Authentication ---

def verify_token(x_api_token: str = Header(None)):
    """Simple token verification - replace with Entra ID for production"""
    if x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    return True

# --- Client APIs ---

@app.post("/register")
async def register_client(registration: ClientRegistration, authorized: bool = Depends(verify_token)):
    """Register a new client with the server"""
    client_id = registration.client_id or f"client-{uuid.uuid4().hex[:8]}"
    
    # Store client info in Redis
    client_info = {
        "client_id": client_id,
        "registered_at": datetime.utcnow().isoformat(),
        "capabilities": json.dumps(registration.capabilities),
        "dataset_labels": json.dumps(registration.dataset_labels)
    }
    redis_client.hset(f"client:{client_id}", mapping=client_info)
    redis_client.sadd("registered_clients", client_id)
    
    return {"client_id": client_id, "status": "registered"}

@app.post("/heartbeat")
async def client_heartbeat(heartbeat: ClientHeartbeat, authorized: bool = Depends(verify_token)):
    """Update client heartbeat"""
    redis_client.hset(f"client:{heartbeat.client_id}", "last_heartbeat", datetime.utcnow().isoformat())
    redis_client.setex(f"heartbeat:{heartbeat.client_id}", 60, heartbeat.status)
    return {"status": "ok"}

@app.get("/assignments")
async def get_assignments(client_id: str, authorized: bool = Depends(verify_token)):
    """Poll for work assignments (outbound-only client communication)"""
    
    # Check for assigned tasks in queue
    assignment_key = f"assignment:{client_id}"
    assignment = redis_client.get(assignment_key)
    
    if not assignment:
        # Check global queue for unassigned tasks
        task_data = redis_client.lpop("task_queue")
        if task_data:
            task_info = json.loads(task_data)
            
            # Assign to this client
            assignment = {
                "run_id": task_info["run_id"],
                "round": task_info.get("round", 1),
                "task_url": f"{PUBLIC_BASE_URL}/storage/tasks/{task_info['run_id']}/task.py",
                "config": task_info.get("config", {}),
                "assigned_at": datetime.utcnow().isoformat()
            }
            
            # Store assignment with TTL
            redis_client.setex(assignment_key, 300, json.dumps(assignment))
            redis_client.hset(f"run:{task_info['run_id']}", "assigned_to", client_id)
            
            return assignment
        else:
            return {}  # No work available
    
    return json.loads(assignment)

@app.post("/upload_update")
async def upload_update(
    run_id: str,
    client_id: str,
    round: int,
    weights_file: UploadFile = File(...),
    metrics: str = Form(...),
    num_examples: int = Form(...),
    authorized: bool = Depends(verify_token)
):
    """Receive weight updates from clients"""
    
    # Save weights file
    update_dir = STORAGE_PATH / "updates" / run_id / f"round_{round}"
    update_dir.mkdir(parents=True, exist_ok=True)
    
    weights_path = update_dir / f"{client_id}_weights.npz"
    content = await weights_file.read()
    with open(weights_path, "wb") as f:
        f.write(content)
    
    # Store update metadata
    update_info = {
        "client_id": client_id,
        "round": round,
        "metrics": metrics,
        "num_examples": num_examples,
        "weights_path": str(weights_path),
        "received_at": datetime.utcnow().isoformat()
    }
    
    # Add to Redis for aggregation
    redis_client.rpush(f"updates:{run_id}:round_{round}", json.dumps(update_info))
    redis_client.hincrby(f"run:{run_id}", "received_updates", 1)
    
    # Clear assignment
    redis_client.delete(f"assignment:{client_id}")
    
    # Trigger aggregation if all updates received
    expected = int(redis_client.hget(f"run:{run_id}", "expected_clients") or 1)
    received = int(redis_client.hget(f"run:{run_id}", "received_updates") or 0)
    
    if received >= expected:
        # Queue aggregation task
        redis_client.rpush("aggregation_queue", json.dumps({
            "run_id": run_id,
            "round": round
        }))
    
    return {"status": "received", "run_id": run_id, "round": round}

# --- Modeller APIs ---

@app.post("/upload")
async def upload_task(
    task: UploadFile = File(...),
    config: Optional[UploadFile] = File(None),
    authorized: bool = Depends(verify_token)
):
    """Primary task submission endpoint"""
    
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    task_dir = STORAGE_PATH / "tasks" / run_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save task.py
    task_content = await task.read()
    task_path = task_dir / "task.py"
    with open(task_path, "wb") as f:
        f.write(task_content)
    
    # Parse config or use defaults
    config_data = {}
    if config:
        config_content = await config.read()
        config_data = yaml.safe_load(config_content)
        config_path = task_dir / "run.yaml"
        with open(config_path, "wb") as f:
            f.write(config_content)
    
    # Set defaults
    config_data.setdefault("run_name", run_id)
    config_data.setdefault("num_rounds", 1)
    config_data.setdefault("num_clients", 1)
    config_data.setdefault("framework", "pytorch")
    
    # Create run metadata
    run_info = {
        "run_id": run_id,
        "status": "queued",
        "task_path": str(task_path),
        "config": json.dumps(config_data),
        "expected_clients": config_data["num_clients"],
        "received_updates": 0,
        "created_at": datetime.utcnow().isoformat()
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
    
    return {"run_id": run_id, "status": "queued"}

@app.get("/runs/{run_id}")
async def get_run_status(run_id: str, authorized: bool = Depends(verify_token)):
    """Get run status and artifacts"""
    
    run_data = redis_client.hgetall(f"run:{run_id}")
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Check for artifacts
    artifacts = {}
    artifact_dir = STORAGE_PATH / "artifacts" / run_id
    if artifact_dir.exists():
        for file in artifact_dir.glob("*"):
            artifacts[file.stem] = f"{PUBLIC_BASE_URL}/artifacts/{run_id}/{file.name}"
    
    # Parse metrics if available
    metrics = None
    metrics_file = artifact_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
    
    expected = int(run_data.get("expected_clients", 1))
    received = int(run_data.get("received_updates", 0))
    
    return RunStatus(
        run_id=run_id,
        status=run_data.get("status", "unknown"),
        expected_clients=expected,
        received_updates=received,
        done=run_data.get("status") == "completed",
        artifacts=artifacts,
        metrics=metrics
    )

@app.get("/artifacts/{run_id}/{filename}")
async def download_artifact(run_id: str, filename: str, authorized: bool = Depends(verify_token)):
    """Download training artifacts"""
    
    file_path = STORAGE_PATH / "artifacts" / run_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    return FileResponse(file_path)

@app.get("/storage/tasks/{run_id}/{filename}")
async def download_task(run_id: str, filename: str, authorized: bool = Depends(verify_token)):
    """Download task files for clients"""
    
    file_path = STORAGE_PATH / "tasks" / run_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Task file not found")
    
    return FileResponse(file_path)

# --- Health & Monitoring ---

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "redis": "disconnected"}
        )

@app.get("/")
async def root():
    """API root"""
    return {
        "service": "FL Platform",
        "version": "1.0.0",
        "docs": f"{PUBLIC_BASE_URL}/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)