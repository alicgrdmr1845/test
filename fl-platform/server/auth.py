"""
Authentication Module - Token and OIDC support
Currently uses static tokens, ready for Entra ID integration
"""

import os
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# Configuration
API_TOKEN = os.getenv("API_TOKEN", "demo-token-123")
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"

# Security scheme
security = HTTPBearer(auto_error=False)


class TokenAuth:
    """Simple token-based authentication"""
    
    @staticmethod
    def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> bool:
        """Verify API token from header"""
        if not ENABLE_AUTH:
            return True
        
        if not credentials:
            raise HTTPException(status_code=401, detail="Missing authentication token")
        
        if credentials.credentials != API_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        return True


# Entra ID / OIDC implementation (for future use)
"""
from fastapi_azure_auth import SingleTenantAzureAuthorizationCodeBearer

class EntraIDAuth:
    '''Azure Entra ID (formerly Azure AD) authentication'''
    
    def __init__(self, tenant_id: str, client_id: str):
        self.azure_scheme = SingleTenantAzureAuthorizationCodeBearer(
            app_client_id=client_id,
            tenant_id=tenant_id,
            scopes={
                f"api://{client_id}/user_impersonation": "User Impersonation",
            }
        )
    
    def verify_token(self, token: str = Depends(azure_scheme)) -> Dict[str, Any]:
        '''Verify Entra ID token and return user info'''
        # Token is automatically validated by the azure_scheme
        return {
            "user_id": token.get("oid"),
            "username": token.get("preferred_username"),
            "roles": token.get("roles", [])
        }
    
    def require_role(self, required_role: str):
        '''Decorator to require specific role'''
        def role_checker(user_info: Dict = Depends(self.verify_token)) -> Dict:
            roles = user_info.get("roles", [])
            if required_role not in roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"Role '{required_role}' required"
                )
            return user_info
        return role_checker


# Usage example for Entra ID:
# 
# auth = EntraIDAuth(
#     tenant_id="your-tenant-id",
#     client_id="your-app-client-id"
# )
# 
# @app.get("/protected")
# def protected_endpoint(user=Depends(auth.verify_token)):
#     return {"message": f"Hello {user['username']}"}
# 
# @app.get("/admin")
# def admin_endpoint(user=Depends(auth.require_role("Admin"))):
#     return {"message": "Admin access granted"}
"""


# Client certificate authentication (for client nodes)
"""
class ClientCertAuth:
    '''Certificate-based authentication for FL clients'''
    
    @staticmethod
    def verify_client_cert(request: Request) -> Optional[str]:
        '''Verify client certificate and return client ID'''
        # This requires TLS termination at the application level
        # or passing cert info from reverse proxy
        
        cert_header = request.headers.get("X-Client-Cert")
        if not cert_header:
            raise HTTPException(status_code=401, detail="Client certificate required")
        
        # Parse and validate certificate
        # Return client_id from certificate CN
        # Implementation depends on cert format and validation requirements
        
        return "client_id_from_cert"
"""


# Factory function
def get_auth_handler(auth_type: str = "token"):
    """Get appropriate authentication handler"""
    if auth_type == "token":
        return TokenAuth()
    # elif auth_type == "entra":
    #     return EntraIDAuth(
    #         tenant_id=os.getenv("AZURE_TENANT_ID"),
    #         client_id=os.getenv("AZURE_CLIENT_ID")
    #     )
    else:
        raise ValueError(f"Unknown auth type: {auth_type}")