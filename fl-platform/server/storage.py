"""
Storage Manager - Abstraction layer for artifact storage
Currently uses local filesystem, easily swappable to Azure Blob Storage
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO
from abc import ABC, abstractmethod
from datetime import datetime, timedelta


class StorageManager(ABC):
    """Abstract base class for storage implementations"""
    
    @abstractmethod
    def store_file(self, container: str, filename: str, content: bytes) -> str:
        """Store a file and return its URL/path"""
        pass
    
    @abstractmethod
    def retrieve_file(self, container: str, filename: str) -> Optional[bytes]:
        """Retrieve file content"""
        pass
    
    @abstractmethod
    def delete_file(self, container: str, filename: str) -> bool:
        """Delete a file"""
        pass
    
    @abstractmethod
    def list_files(self, container: str, prefix: str = "") -> list:
        """List files in container with optional prefix"""
        pass
    
    @abstractmethod
    def get_url(self, container: str, filename: str, expiry_hours: int = 24) -> str:
        """Get accessible URL for file"""
        pass


class LocalStorageManager(StorageManager):
    """Local filesystem storage implementation"""
    
    def __init__(self, base_path: str = "/app/storage", base_url: str = "http://localhost:8000"):
        self.base_path = Path(base_path)
        self.base_url = base_url.rstrip('/')
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def store_file(self, container: str, filename: str, content: bytes) -> str:
        """Store file locally"""
        container_path = self.base_path / container
        container_path.mkdir(parents=True, exist_ok=True)
        
        file_path = container_path / filename
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return str(file_path)
    
    def retrieve_file(self, container: str, filename: str) -> Optional[bytes]:
        """Retrieve file from local storage"""
        file_path = self.base_path / container / filename
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return f.read()
        return None
    
    def delete_file(self, container: str, filename: str) -> bool:
        """Delete local file"""
        file_path = self.base_path / container / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def list_files(self, container: str, prefix: str = "") -> list:
        """List files in directory"""
        container_path = self.base_path / container
        if not container_path.exists():
            return []
        
        files = []
        for file_path in container_path.glob(f"{prefix}*"):
            if file_path.is_file():
                files.append(file_path.name)
        return files
    
    def get_url(self, container: str, filename: str, expiry_hours: int = 24) -> str:
        """Get URL for local file (no SAS for local storage)"""
        return f"{self.base_url}/storage/{container}/{filename}"


# Azure Blob Storage implementation (for future use)
"""
class AzureBlobStorageManager(StorageManager):
    '''Azure Blob Storage implementation'''
    
    def __init__(self, connection_string: str, container_name: str = "fl-artifacts"):
        from azure.storage.blob import BlobServiceClient
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
        
        # Ensure container exists
        try:
            self.blob_service.create_container(container_name)
        except:
            pass  # Container already exists
    
    def store_file(self, container: str, filename: str, content: bytes) -> str:
        '''Store file in Azure Blob'''
        blob_name = f"{container}/{filename}"
        blob_client = self.blob_service.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        blob_client.upload_blob(content, overwrite=True)
        return blob_name
    
    def retrieve_file(self, container: str, filename: str) -> Optional[bytes]:
        '''Retrieve file from Azure Blob'''
        blob_name = f"{container}/{filename}"
        try:
            blob_client = self.blob_service.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            return blob_client.download_blob().readall()
        except:
            return None
    
    def delete_file(self, container: str, filename: str) -> bool:
        '''Delete blob'''
        blob_name = f"{container}/{filename}"
        try:
            blob_client = self.blob_service.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            blob_client.delete_blob()
            return True
        except:
            return False
    
    def list_files(self, container: str, prefix: str = "") -> list:
        '''List blobs with prefix'''
        full_prefix = f"{container}/{prefix}"
        blobs = self.blob_service.get_container_client(
            self.container_name
        ).list_blobs(name_starts_with=full_prefix)
        
        files = []
        for blob in blobs:
            # Remove container prefix from name
            name = blob.name.replace(f"{container}/", "")
            files.append(name)
        return files
    
    def get_url(self, container: str, filename: str, expiry_hours: int = 24) -> str:
        '''Generate SAS URL for blob'''
        from azure.storage.blob import generate_blob_sas, BlobSasPermissions
        
        blob_name = f"{container}/{filename}"
        sas_token = generate_blob_sas(
            account_name=self.blob_service.account_name,
            container_name=self.container_name,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
        )
        
        blob_url = f"https://{self.blob_service.account_name}.blob.core.windows.net/"
        return f"{blob_url}{self.container_name}/{blob_name}?{sas_token}"
"""


# Factory function
def create_storage_manager(storage_type: str = "local", **kwargs) -> StorageManager:
    """Factory to create appropriate storage manager"""
    if storage_type == "local":
        return LocalStorageManager(**kwargs)
    # elif storage_type == "azure":
    #     return AzureBlobStorageManager(**kwargs)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")