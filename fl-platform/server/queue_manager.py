"""
Queue Manager - Abstraction layer for job queue
Currently uses Redis, easily swappable to Azure Service Bus
"""

import json
import time
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

import redis


class QueueManager(ABC):
    """Abstract base class for queue implementations"""
    
    @abstractmethod
    def push(self, queue_name: str, item: Dict[str, Any]) -> bool:
        """Add item to queue"""
        pass
    
    @abstractmethod
    def pop(self, queue_name: str, timeout: int = 0) -> Optional[Dict[str, Any]]:
        """Remove and return item from queue"""
        pass
    
    @abstractmethod
    def peek(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """View next item without removing"""
        pass
    
    @abstractmethod
    def length(self, queue_name: str) -> int:
        """Get queue length"""
        pass


class RedisQueueManager(QueueManager):
    """Redis-based queue implementation"""
    
    def __init__(self, host: str = "localhost", port: int = 6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
    
    def push(self, queue_name: str, item: Dict[str, Any]) -> bool:
        try:
            self.client.rpush(queue_name, json.dumps(item))
            return True
        except Exception as e:
            print(f"Queue push failed: {e}")
            return False
    
    def pop(self, queue_name: str, timeout: int = 0) -> Optional[Dict[str, Any]]:
        try:
            if timeout > 0:
                result = self.client.blpop(queue_name, timeout=timeout)
                if result:
                    return json.loads(result[1])
            else:
                result = self.client.lpop(queue_name)
                if result:
                    return json.loads(result)
            return None
        except Exception as e:
            print(f"Queue pop failed: {e}")
            return None
    
    def peek(self, queue_name: str) -> Optional[Dict[str, Any]]:
        try:
            result = self.client.lindex(queue_name, 0)
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            print(f"Queue peek failed: {e}")
            return None
    
    def length(self, queue_name: str) -> int:
        try:
            return self.client.llen(queue_name)
        except Exception:
            return 0


# Azure Service Bus implementation (for future use)
"""
class AzureServiceBusQueueManager(QueueManager):
    '''Azure Service Bus queue implementation'''
    
    def __init__(self, connection_string: str, queue_name: str):
        from azure.servicebus import ServiceBusClient
        self.client = ServiceBusClient.from_connection_string(connection_string)
        self.queue_name = queue_name
        self.sender = self.client.get_queue_sender(queue_name)
        self.receiver = self.client.get_queue_receiver(queue_name)
    
    def push(self, queue_name: str, item: Dict[str, Any]) -> bool:
        from azure.servicebus import ServiceBusMessage
        try:
            message = ServiceBusMessage(json.dumps(item))
            self.sender.send_messages(message)
            return True
        except Exception as e:
            print(f"Azure SB push failed: {e}")
            return False
    
    def pop(self, queue_name: str, timeout: int = 0) -> Optional[Dict[str, Any]]:
        try:
            messages = self.receiver.receive_messages(max_wait_time=timeout or 1)
            if messages:
                message = messages[0]
                data = json.loads(str(message))
                self.receiver.complete_message(message)
                return data
            return None
        except Exception as e:
            print(f"Azure SB pop failed: {e}")
            return None
    
    # Implement other methods similarly...
"""


# Factory function
def create_queue_manager(queue_type: str = "redis", **kwargs) -> QueueManager:
    """Factory to create appropriate queue manager"""
    if queue_type == "redis":
        return RedisQueueManager(**kwargs)
    # elif queue_type == "azure":
    #     return AzureServiceBusQueueManager(**kwargs)
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")