# server/api_flower.py
"""
Updated FastAPI Server with Flower Integration - FIXED VERSION
"""

import os
import json
import uuid
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# Added imports for visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Docker
import matplotlib.pyplot as plt

from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
import redis
import numpy as np
import logging

# Import Flower orchestrator
from server.flower_orchestrator import FlowerOrchestrator

app = FastAPI(title="FL Platform with Flower", version="2.0.0")

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
API_TOKEN = os.getenv("API_TOKEN", "demo-token-123")
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "/app/storage"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
MAX_CONCURRENT_RUNS = int(os.getenv("MAX_CONCURRENT_RUNS", 5))

# Ensure storage directories exist
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
(STORAGE_PATH / "tasks").mkdir(exist_ok=True)
(STORAGE_PATH / "artifacts").mkdir(exist_ok=True)

# Initialize Redis connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Initialize Flower orchestrator
flower_orchestrator = FlowerOrchestrator(
    redis_client=redis_client,
    storage_path=STORAGE_PATH,
    max_concurrent_runs=MAX_CONCURRENT_RUNS
)

# --- Pydantic Models ---

class ClientRegistration(BaseModel):
    client_id: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = {}
    dataset_labels: Optional[List[str]] = []
    data_path: Optional[str] = None

class RunSubmission(BaseModel):
    run_name: Optional[str] = None
    num_rounds: int = 1
    num_clients: int = 1
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    framework: str = "pytorch"
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0

class RunStatus(BaseModel):
    run_id: str
    status: str
    flower_server: Optional[str]
    current_round: Optional[int]
    expected_clients: int
    clients_participated: Optional[int]
    created_at: str
    metrics: Optional[Dict[str, Any]]
    artifacts: Optional[Dict[str, str]]

# --- Authentication ---

def verify_token(x_api_token: str = Header(None)):
    """Simple token verification"""
    if x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    return True

# --- Client Management APIs ---

@app.post("/register", response_model=Dict[str, str])
async def register_client(
    registration: ClientRegistration,
    authorized: bool = Depends(verify_token)
):
    """Register a new FL client"""
    
    client_id = registration.client_id or f"client-{uuid.uuid4().hex[:8]}"
    
    # Store client info in Redis
    client_info = {
        "client_id": client_id,
        "registered_at": datetime.utcnow().isoformat(),
        "capabilities": json.dumps(registration.capabilities),
        "dataset_labels": json.dumps(registration.dataset_labels),
        "data_path": registration.data_path or f"/data/{client_id}",
        "status": "idle"
    }
    
    redis_client.hset(f"client:{client_id}", mapping=client_info)
    redis_client.sadd("registered_clients", client_id)
    
    # Set TTL for client registration (24 hours)
    redis_client.expire(f"client:{client_id}", 86400)
    
    return {"client_id": client_id, "status": "registered"}

@app.get("/assignments")
async def get_assignments(
    client_id: str,
    authorized: bool = Depends(verify_token)
):
    """
    Poll for work assignments (outbound-only client communication)
    Returns task assignment if available
    """
    
    # Update client heartbeat
    redis_client.hset(f"client:{client_id}", "last_seen", datetime.utcnow().isoformat())
    
    # Check for existing assignment
    assignment_key = f"assignment:{client_id}"
    existing_assignment = redis_client.get(assignment_key)
    
    if existing_assignment:
        return json.loads(existing_assignment)
    
    # Check task queue for new work
    task_json = redis_client.lpop("task_queue")
    
    if task_json:
        task_info = json.loads(task_json)
        run_id = task_info["run_id"]
        
        # Get run details
        run_data = redis_client.hgetall(f"run:{run_id}")
        
        if run_data:
            # Create assignment
            assignment = {
                "run_id": run_id,
                "task_url": f"{PUBLIC_BASE_URL}/storage/tasks/{run_id}/task.py",
                "flower_server": task_info.get("flower_server", "0.0.0.0:9093"),
                "config": json.loads(run_data.get("config", "{}")),
                "assigned_at": datetime.utcnow().isoformat()
            }
            
            # Store assignment with TTL (5 minutes)
            redis_client.setex(
                assignment_key,
                300,
                json.dumps(assignment)
            )
            
            # Track assignment
            redis_client.sadd(f"run:{run_id}:assigned_clients", client_id)
            
            return assignment
    
    # No work available
    return {}

@app.post("/heartbeat")
async def client_heartbeat(
    client_id: str,
    status: str = "healthy",
    authorized: bool = Depends(verify_token)
):
    """Update client heartbeat and status"""
    
    redis_client.hset(f"client:{client_id}", mapping={
        "last_heartbeat": datetime.utcnow().isoformat(),
        "status": status
    })
    
    # Refresh TTL
    redis_client.expire(f"client:{client_id}", 86400)
    
    return {"status": "ok"}

@app.post("/clear_assignment")
async def clear_assignment(
    request: Dict[str, str],
    authorized: bool = Depends(verify_token)
):
    """Clear a completed assignment"""
    client_id = request.get("client_id")
    run_id = request.get("run_id")
    
    if not client_id or not run_id:
        raise HTTPException(status_code=400, detail="Missing client_id or run_id")
    
    # Clear the assignment
    assignment_key = f"assignment:{client_id}"
    redis_client.delete(assignment_key)
    
    # Remove from assigned clients
    redis_client.srem(f"run:{run_id}:assigned_clients", client_id)
    
    logger.info(f"Cleared assignment for client {client_id}, run {run_id}")
    
    return {"status": "cleared"}

# --- Task Submission APIs ---

# Fix for the upload endpoint in server/api_flower.py
# Replace the upload_task function (around line 200) with this:

@app.post("/upload", response_model=Dict[str, str])
async def upload_task(
    background_tasks: BackgroundTasks,
    task: UploadFile = File(...),
    config: Optional[UploadFile] = File(None),
    authorized: bool = Depends(verify_token)
):
    """
    Upload and start FL training task
    FIXED: Queue tasks BEFORE starting Flower server
    """
    
    # Generate run ID
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    
    # Create task directory
    task_dir = STORAGE_PATH / "tasks" / run_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save task.py
    task_content = await task.read()
    task_path = task_dir / "task.py"
    with open(task_path, "wb") as f:
        f.write(task_content)
    
    # Parse configuration
    config_data = {}
    if config:
        config_content = await config.read()
        config_data = yaml.safe_load(config_content)
        
        # Save config for reference
        config_path = task_dir / "run.yaml"
        with open(config_path, "wb") as f:
            f.write(config_content)
    
    # Set defaults
    config_data.setdefault("run_name", run_id)
    config_data.setdefault("num_rounds", 3)  # Changed from 1 to 3
    config_data.setdefault("num_clients", 1)
    config_data.setdefault("local_epochs", 5)
    config_data.setdefault("batch_size", 32)
    config_data.setdefault("learning_rate", 0.01)
    config_data.setdefault("framework", "pytorch")
    
    # Store run metadata in Redis
    run_info = {
        "run_id": run_id,
        "status": "initializing",
        "task_path": str(task_path),
        "config": json.dumps(config_data),
        "expected_clients": config_data["num_clients"],
        "created_at": datetime.utcnow().isoformat()
    }
    
    redis_client.hset(f"run:{run_id}", mapping=run_info)
    redis_client.rpush("all_runs", run_id)
    
    # FIXED: Queue tasks for clients FIRST, before starting Flower server
    # Use the actual hostname that clients can reach
    flower_server_address = f"fl-platform:9093"  # This will be updated after server starts
    
    for i in range(config_data["num_clients"]):
        task_item = {
            "run_id": run_id,
            "flower_server": flower_server_address  # Placeholder, will be updated
        }
        redis_client.rpush("task_queue", json.dumps(task_item))
        logger.info(f"Queued task for client {i+1}/{config_data['num_clients']}")
    
    logger.info(f"Queued {config_data['num_clients']} tasks for run {run_id}")
    
    try:
        # NOW start Flower server (it will wait for clients that are polling)
        flower_info = flower_orchestrator.start_training_run(
            run_id=run_id,
            task_path=str(task_path),
            config=config_data
        )
        
        # Update the queued tasks with the actual server address
        # Get all tasks from queue, update the ones for this run, and put them back
        temp_queue = []
        queue_len = redis_client.llen("task_queue")
        
        for _ in range(queue_len):
            task_json = redis_client.lpop("task_queue")
            if task_json:
                task_data = json.loads(task_json)
                if task_data["run_id"] == run_id:
                    # Update with actual server address
                    public_hostname = urlparse(PUBLIC_BASE_URL).hostname
                    task_data["flower_server"] = flower_info["flower_server"].replace("0.0.0.0", public_hostname)
                temp_queue.append(json.dumps(task_data))
        
        # Put all tasks back
        for task_json in temp_queue:
            redis_client.rpush("task_queue", task_json)
        
        return {
            "run_id": run_id,
            "status": "training_started",
            "flower_server": flower_info["flower_server"].replace("0.0.0.0", "fl-platform")
        }
        
    except Exception as e:
        # On failure, clear the queued tasks
        # Remove tasks for this run from the queue
        temp_queue = []
        queue_len = redis_client.llen("task_queue")
        
        for _ in range(queue_len):
            task_json = redis_client.lpop("task_queue")
            if task_json:
                task_data = json.loads(task_json)
                if task_data["run_id"] != run_id:  # Keep tasks not for this run
                    temp_queue.append(task_json)
        
        # Put back tasks not for this run
        for task_json in temp_queue:
            redis_client.rpush("task_queue", task_json)
        
        # Update status on failure
        redis_client.hset(f"run:{run_id}", mapping={
            "status": "failed",
            "error": str(e)
        })
        
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.post("/submit_run", response_model=Dict[str, str])
async def submit_run(
    submission: RunSubmission,
    task: UploadFile = File(...),
    authorized: bool = Depends(verify_token)
):
    """
    Alternative API for submitting runs with JSON config
    """
    
    # Generate run ID
    run_id = submission.run_name or f"run_{uuid.uuid4().hex[:8]}"
    
    # Create task directory
    task_dir = STORAGE_PATH / "tasks" / run_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save task file
    task_content = await task.read()
    task_path = task_dir / "task.py"
    with open(task_path, "wb") as f:
        f.write(task_content)
    
    # Convert submission to config dict
    config_data = submission.dict()
    config_data["run_name"] = run_id
    
    # Store in Redis
    run_info = {
        "run_id": run_id,
        "status": "initializing",
        "task_path": str(task_path),
        "config": json.dumps(config_data),
        "expected_clients": config_data["num_clients"],
        "created_at": datetime.utcnow().isoformat()
    }
    
    redis_client.hset(f"run:{run_id}", mapping=run_info)
    
    try:
        # Start Flower server
        flower_info = flower_orchestrator.start_training_run(
            run_id=run_id,
            task_path=str(task_path),
            config=config_data
        )
        
        # Queue tasks for clients
        for i in range(config_data["num_clients"]):
            redis_client.rpush("task_queue", json.dumps({
                "run_id": run_id,
                "flower_server": flower_info["flower_server"].replace("0.0.0.0", "fl-platform")
            }))
        
        return {
            "run_id": run_id,
            "status": "training_started",
            "flower_server": flower_info["flower_server"].replace("0.0.0.0", "fl-platform")
        }
        
    except Exception as e:
        redis_client.hset(f"run:{run_id}", "status", "failed")
        raise HTTPException(status_code=500, detail=str(e))

# --- Run Monitoring APIs ---

@app.get("/runs/{run_id}", response_model=RunStatus)
async def get_run_status(
    run_id: str,
    authorized: bool = Depends(verify_token)
):
    """Get detailed run status including Flower metrics"""
    
    # Get run data from Redis
    run_data = redis_client.hgetall(f"run:{run_id}")
    
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Get Flower server status
    flower_status = flower_orchestrator.get_run_status(run_id)
    
    # Load metrics from storage
    metrics = None
    artifacts = {}
    
    artifact_dir = STORAGE_PATH / "artifacts" / run_id
    if artifact_dir.exists():
        # Load metrics
        metrics_file = artifact_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
        
        # List artifacts
        for file_path in artifact_dir.glob("*.npz"):
            artifacts[file_path.stem] = f"{PUBLIC_BASE_URL}/artifacts/{run_id}/{file_path.name}"
        
        if metrics_file.exists():
            artifacts["metrics"] = f"{PUBLIC_BASE_URL}/artifacts/{run_id}/metrics.json"
    
    # Build response
    return RunStatus(
        run_id=run_id,
        status=run_data.get("status", "unknown"),
        flower_server=flower_status["flower_server"] if flower_status else None,
        current_round=int(run_data.get("current_round", 0)),
        expected_clients=int(run_data.get("expected_clients", 1)),
        clients_participated=int(run_data.get("clients_participated", 0)),
        created_at=run_data.get("created_at", ""),
        metrics=metrics,
        artifacts=artifacts
    )

@app.get("/runs", response_model=List[Dict[str, Any]])
async def list_runs(
    status: Optional[str] = None,
    limit: int = 100,
    authorized: bool = Depends(verify_token)
):
    """List all runs with optional status filter"""
    
    # Get all run IDs
    run_ids = redis_client.lrange("all_runs", 0, limit - 1)
    
    runs = []
    for run_id in run_ids:
        run_data = redis_client.hgetall(f"run:{run_id}")
        if run_data:
            # Apply status filter if provided
            if status and run_data.get("status") != status:
                continue
            
            runs.append({
                "run_id": run_id,
                "status": run_data.get("status"),
                "created_at": run_data.get("created_at"),
                "expected_clients": int(run_data.get("expected_clients", 0)),
                "current_round": int(run_data.get("current_round", 0))
            })
    
    return runs

@app.delete("/runs/{run_id}")
async def stop_run(
    run_id: str,
    authorized: bool = Depends(verify_token)
):
    """Stop a running training job"""
    
    # Check if run exists
    run_data = redis_client.hgetall(f"run:{run_id}")
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Stop Flower server
    stopped = flower_orchestrator.stop_training_run(run_id)
    
    if stopped:
        # Update Redis
        redis_client.hset(f"run:{run_id}", "status", "cancelled")
        
        # Clear any pending assignments
        redis_client.delete(f"run:{run_id}:assigned_clients")
        
        return {"status": "stopped", "run_id": run_id}
    else:
        return {"status": "not_running", "run_id": run_id}


# --- NEW: Visualization Endpoints ---

@app.get("/visualize/{run_id}")
async def simple_visualize(run_id: str, authorized: bool = Depends(verify_token)):
    """Generate simple text visualization for completed run"""
    
    # Get run data
    run_data = redis_client.hgetall(f"run:{run_id}")
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Load metrics
    metrics_file = STORAGE_PATH / "artifacts" / run_id / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="No metrics found")
    
    with open(metrics_file) as f:
        metrics_data = json.load(f)
    
    # Create simple text summary
    history = metrics_data.get('history', {})
    val_loss_tuples = history.get('val_loss', [])
    
    summary = f"""Training Results for {run_id}
=====================================
Status: {run_data.get('status')}
Rounds: {metrics_data.get('current_round', 0)}
Validation Loss Progress:
"""
    
    val_loss = [loss for _, loss in val_loss_tuples]

    for i, loss in enumerate(val_loss, 1):
        summary += f"  Round {i}: {loss:.4f}\n"
    
    if val_loss:
        improvement = val_loss[0] - val_loss[-1]
        summary += f"\nTotal improvement: {improvement:.4f}"
    
    return {"summary": summary}

@app.get("/runs/{run_id}/plot")
async def get_training_plot(run_id: str, authorized: bool = Depends(verify_token)):
    """Get training plot for a run (auto-generate if needed)"""
    return await simple_visualize(run_id)

@app.get("/plot/{run_id}")
async def plot_metrics(run_id: str, authorized: bool = Depends(verify_token)):
    """Generate plot with loss and accuracy graphs"""
    
    # Check if metrics file exists
    metrics_file = STORAGE_PATH / "artifacts" / run_id / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail=f"Metrics not found for run {run_id}")
    
    # Load metrics
    with open(metrics_file) as f:
        data = json.load(f)
    
    # Import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Extract data - handle tuple format [[round, value], ...]
    history = data.get('history', {})
    
    # Extract values from [round, value] tuples
    def get_values(metric_list):
        if not metric_list:
            return []
        if isinstance(metric_list[0], (list, tuple)):
            return [v for _, v in metric_list]
        return metric_list
    
    train_loss = get_values(history.get('train_loss', []))
    train_acc = get_values(history.get('train_accuracy', []))
    val_loss = get_values(history.get('val_loss', []))
    val_acc = get_values(history.get('val_accuracy', []))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Loss
    rounds = list(range(1, len(train_loss) + 1)) if train_loss else []
    if train_loss:
        ax1.plot(rounds, train_loss, 'b-o', label='Train Loss', linewidth=2)
    if val_loss:
        rounds = list(range(1, len(val_loss) + 1))
        ax1.plot(rounds, val_loss, 'r-o', label='Val Loss', linewidth=2)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss per Round')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Accuracy
    if train_acc:
        rounds = list(range(1, len(train_acc) + 1))
        # Convert to percentage
        train_acc_pct = [a * 100 for a in train_acc]
        ax2.plot(rounds, train_acc_pct, 'b-o', label='Train Accuracy', linewidth=2)
    if val_acc:
        rounds = list(range(1, len(val_acc) + 1))
        # Convert to percentage
        val_acc_pct = [a * 100 for a in val_acc]
        ax2.plot(rounds, val_acc_pct, 'r-o', label='Val Accuracy', linewidth=2)
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy per Round')
    ax2.legend()
    ax2.grid(True)
    
    # Overall title
    fig.suptitle(f'Training Metrics for {run_id}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plot_path = STORAGE_PATH / "artifacts" / run_id / "plot.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Return the image file
    return FileResponse(plot_path, media_type="image/png")

# --- Artifact Download APIs ---

@app.get("/artifacts/{run_id}/{filename}")
async def download_artifact(
    run_id: str,
    filename: str,
    authorized: bool = Depends(verify_token)
):
    """Download training artifacts"""
    
    file_path = STORAGE_PATH / "artifacts" / run_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@app.get("/storage/tasks/{run_id}/{filename}")
async def download_task(
    run_id: str,
    filename: str,
    authorized: bool = Depends(verify_token)
):
    """Download task files for clients"""
    
    file_path = STORAGE_PATH / "tasks" / run_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Task file not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/plain" if filename.endswith(".py") else "application/octet-stream"
    )

# --- Client Monitoring APIs ---

@app.get("/clients", response_model=List[Dict[str, Any]])
async def list_clients(
    authorized: bool = Depends(verify_token)
):
    """List all registered clients"""
    
    client_ids = redis_client.smembers("registered_clients")
    
    clients = []
    for client_id in client_ids:
        client_data = redis_client.hgetall(f"client:{client_id}")
        if client_data:
            clients.append({
                "client_id": client_id,
                "status": client_data.get("status", "unknown"),
                "last_seen": client_data.get("last_seen"),
                "registered_at": client_data.get("registered_at"),
                "dataset_labels": json.loads(client_data.get("dataset_labels", "[]"))
            })
    
    return clients

@app.get("/clients/{client_id}")
async def get_client_info(
    client_id: str,
    authorized: bool = Depends(verify_token)
):
    """Get detailed client information"""
    
    client_data = redis_client.hgetall(f"client:{client_id}")
    
    if not client_data:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Get current assignment if any
    assignment_key = f"assignment:{client_id}"
    assignment = redis_client.get(assignment_key)
    
    return {
        "client_id": client_id,
        "status": client_data.get("status"),
        "last_seen": client_data.get("last_seen"),
        "registered_at": client_data.get("registered_at"),
        "capabilities": json.loads(client_data.get("capabilities", "{}")),
        "dataset_labels": json.loads(client_data.get("dataset_labels", "[]")),
        "current_assignment": json.loads(assignment) if assignment else None
    }

# --- Health & System APIs ---

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Check Redis
    try:
        redis_client.ping()
        health_status["redis"] = "connected"
    except:
        health_status["redis"] = "disconnected"
        health_status["status"] = "degraded"
    
    # Check Flower orchestrator
    active_runs = len(flower_orchestrator.active_runs)
    health_status["active_runs"] = active_runs
    health_status["max_runs"] = flower_orchestrator.max_concurrent_runs
    
    return health_status

@app.get("/stats")
async def get_statistics(
    authorized: bool = Depends(verify_token)
):
    """Get platform statistics"""
    
    # Count entities
    total_runs = redis_client.llen("all_runs")
    total_clients = redis_client.scard("registered_clients")
    pending_tasks = redis_client.llen("task_queue")
    
    # Count runs by status
    run_stats = {
        "total": total_runs,
        "active": len(flower_orchestrator.active_runs),
        "completed": 0,
        "failed": 0
    }
    
    # Sample some runs to get status counts
    run_ids = redis_client.lrange("all_runs", 0, 99)
    for run_id in run_ids:
        status = redis_client.hget(f"run:{run_id}", "status")
        if status == "completed":
            run_stats["completed"] += 1
        elif status == "failed":
            run_stats["failed"] += 1
    
    return {
        "runs": run_stats,
        "clients": {
            "total": total_clients,
            "online": 0  # Would need to check heartbeats
        },
        "queue": {
            "pending_tasks": pending_tasks
        },
        "system": {
            "active_flower_servers": len(flower_orchestrator.active_runs),
            "max_concurrent_runs": flower_orchestrator.max_concurrent_runs
        }
    }

@app.get("/")
async def root():
    """API root with documentation links"""
    
    return {
        "service": "FL Platform with Flower",
        "version": "2.0.0",
        "endpoints": {
            "docs": f"{PUBLIC_BASE_URL}/docs",
            "health": f"{PUBLIC_BASE_URL}/health",
            "stats": f"{PUBLIC_BASE_URL}/stats"
        },
        "features": [
            "Dynamic task dispatch with Flower",
            "Polling-based client communication",
            "Multiple concurrent training runs",
            "Redis-backed state management",
            "Artifact storage and retrieval"
        ]
    }

# --- WebSocket Support (Optional) ---

from fastapi import WebSocket, WebSocketDisconnect
import asyncio

@app.websocket("/ws/runs/{run_id}")
async def websocket_run_monitor(
    websocket: WebSocket,
    run_id: str
):
    """WebSocket endpoint for real-time run monitoring"""
    
    await websocket.accept()
    
    try:
        while True:
            # Get run status
            run_data = redis_client.hgetall(f"run:{run_id}")
            
            if run_data:
                # Get current metrics
                artifact_dir = STORAGE_PATH / "artifacts" / run_id
                metrics_file = artifact_dir / "metrics.json"
                
                status_update = {
                    "run_id": run_id,
                    "status": run_data.get("status"),
                    "current_round": int(run_data.get("current_round", 0)),
                    "timestamp": time.time()
                }
                
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        status_update["metrics"] = json.load(f)
                
                await websocket.send_json(status_update)
            
            # Wait before next update
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close()

# DO NOT add the if __name__ == "__main__" block with uvicorn.run()
# The server will be started via the Docker CMD or command line# Server will be started via Docker CMD

# Add this temporary debug function at the end of the file
import sys
print("API_FLOWER.PY LOADED", file=sys.stderr)
