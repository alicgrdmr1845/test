"""
FL Controller - Handles aggregation using FedAvg
Polls aggregation queue and performs federated averaging
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import redis
import numpy as np

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "/app/storage"))

# Redis connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


class FederatedAggregator:
    """Simple FedAvg implementation"""
    
    @staticmethod
    def aggregate_weights(updates: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """
        Perform Federated Averaging on client updates
        
        Args:
            updates: List of client updates with weights and metrics
            
        Returns:
            Aggregated weights and metrics
        """
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Load weights from each client
        all_weights = []
        all_num_examples = []
        all_metrics = []
        
        for update in updates:
            # Load weights file
            weights_data = np.load(update["weights_path"], allow_pickle=True)
            
            # Extract weights arrays (assumes they're stored as 'weight_0', 'weight_1', etc.)
            client_weights = []
            idx = 0
            while f'weight_{idx}' in weights_data:
                client_weights.append(weights_data[f'weight_{idx}'])
                idx += 1
            
            # If no indexed weights, try to load 'weights' key
            if not client_weights and 'weights' in weights_data:
                client_weights = weights_data['weights'].tolist()
            
            all_weights.append(client_weights)
            all_num_examples.append(update["num_examples"])
            
            # Parse metrics
            metrics = json.loads(update["metrics"]) if isinstance(update["metrics"], str) else update["metrics"]
            all_metrics.append(metrics)
        
        # Perform weighted averaging
        total_examples = sum(all_num_examples)
        aggregated_weights = []
        
        # Average each layer's weights
        num_layers = len(all_weights[0])
        for layer_idx in range(num_layers):
            # Weighted sum for this layer
            layer_sum = None
            for client_idx, client_weights in enumerate(all_weights):
                weight = all_num_examples[client_idx] / total_examples
                if layer_sum is None:
                    layer_sum = client_weights[layer_idx] * weight
                else:
                    layer_sum += client_weights[layer_idx] * weight
            aggregated_weights.append(layer_sum)
        
        # Average metrics
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            values = [m.get(key, 0) for m in all_metrics]
            weights = [n / total_examples for n in all_num_examples]
            aggregated_metrics[key] = sum(v * w for v, w in zip(values, weights))
        
        aggregated_metrics["total_examples"] = total_examples
        aggregated_metrics["num_clients"] = len(updates)
        
        return aggregated_weights, aggregated_metrics


def process_aggregation_task(task_data: Dict[str, Any]):
    """Process a single aggregation task"""
    
    run_id = task_data["run_id"]
    round_num = task_data["round"]
    
    logger.info(f"Starting aggregation for {run_id} round {round_num}")
    
    try:
        # Get all updates for this round
        updates_key = f"updates:{run_id}:round_{round_num}"
        raw_updates = redis_client.lrange(updates_key, 0, -1)
        
        if not raw_updates:
            logger.warning(f"No updates found for {run_id} round {round_num}")
            return
        
        # Parse updates
        updates = [json.loads(u) for u in raw_updates]
        logger.info(f"Aggregating {len(updates)} updates")
        
        # Perform aggregation
        aggregator = FederatedAggregator()
        aggregated_weights, aggregated_metrics = aggregator.aggregate_weights(updates)
        
        # Save aggregated weights
        artifact_dir = STORAGE_PATH / "artifacts" / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        weights_file = artifact_dir / f"aggregated_round_{round_num}.npz"
        
        # Save weights with proper keys
        save_dict = {f'weight_{i}': w for i, w in enumerate(aggregated_weights)}
        np.savez_compressed(weights_file, **save_dict)
        
        # Save metrics
        metrics_file = artifact_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                f"round_{round_num}": aggregated_metrics,
                "final_metrics": aggregated_metrics  # For single-round runs
            }, f, indent=2)
        
        # Update run status
        redis_client.hset(f"run:{run_id}", "status", "completed")
        redis_client.hset(f"run:{run_id}", "aggregated_at", time.time())
        
        logger.info(f"Aggregation completed for {run_id} round {round_num}")
        logger.info(f"Metrics: {aggregated_metrics}")
        
        # Check if we need more rounds (multi-round support)
        run_data = redis_client.hgetall(f"run:{run_id}")
        config = json.loads(run_data.get("config", "{}"))
        num_rounds = config.get("num_rounds", 1)
        
        if round_num < num_rounds:
            # Queue next round (uncomment to enable)
            # next_task = {
            #     "run_id": run_id,
            #     "round": round_num + 1,
            #     "config": config
            # }
            # redis_client.rpush("task_queue", json.dumps(next_task))
            # redis_client.hset(f"run:{run_id}", "status", "round_" + str(round_num + 1))
            # redis_client.hset(f"run:{run_id}", "received_updates", 0)
            logger.info(f"Multi-round training: Round {round_num + 1} would start here")
        
    except Exception as e:
        logger.error(f"Aggregation failed for {run_id}: {str(e)}")
        redis_client.hset(f"run:{run_id}", "status", "failed")
        redis_client.hset(f"run:{run_id}", "error", str(e))


def main():
    """Main aggregation loop"""
    
    logger.info("FL Aggregator started")
    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    
    while True:
        try:
            # Poll aggregation queue (blocking pop with 5 second timeout)
            result = redis_client.blpop("aggregation_queue", timeout=5)
            
            if result:
                _, task_json = result
                task_data = json.loads(task_json)
                process_aggregation_task(task_data)
            
        except KeyboardInterrupt:
            logger.info("Aggregator shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in aggregation loop: {str(e)}")
            time.sleep(5)  # Wait before retrying


if __name__ == "__main__":
    main()