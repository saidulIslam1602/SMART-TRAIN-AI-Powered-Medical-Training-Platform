"""
Advanced Production API for SMART-TRAIN platform v2.

This module implements enterprise-grade API features including:
- Advanced authentication and RBAC
- Rate limiting and request throttling
- A/B testing framework
- Advanced caching and optimization
- Comprehensive monitoring and analytics
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import redis
from fastapi import FastAPI, HTTPException, Depends, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from smart_train.core.config import get_config
from smart_train.core.logging import setup_logging, get_logger
from smart_train.models.cpr_quality_model import CPRQualityAssessmentModel
from smart_train.models.realtime_feedback import RealTimeFeedbackModel
from smart_train.monitoring.analytics_dashboard import MetricsCollector, MetricType
from smart_train.compliance.audit_trail import AuditTrailManager, AuditEventType, AuditSeverity

# Initialize configuration and logging
config = get_config()
setup_logging(log_level='INFO')
logger = get_logger(__name__)

# Initialize Redis for caching and rate limiting
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")

# Initialize metrics collector
metrics_collector = MetricsCollector(config=None)  # Will be properly configured

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration', ['endpoint'])
ACTIVE_CONNECTIONS = Gauge('api_active_connections', 'Active connections')
MODEL_INFERENCE_COUNT = Counter('model_inferences_total', 'Model inferences', ['model_name'])


# Pydantic models for request/response validation
class AnalysisRequest(BaseModel):
    """Request model for analysis endpoints."""
    data: List[List[List[float]]] = Field(..., description="Pose landmarks data")
    session_id: str = Field(..., description="Training session ID")
    user_id: str = Field(..., description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    
    @validator('data')
    def validate_pose_data(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Pose data cannot be empty")
        
        # Validate pose data structure (sequence_length, 33, 3)
        for frame in v:
            if len(frame) != 33:
                raise ValueError("Each frame must have 33 pose landmarks")
            for landmark in frame:
                if len(landmark) != 3:
                    raise ValueError("Each landmark must have 3 coordinates (x, y, z)")
        
        return v


class AnalysisResponse(BaseModel):
    """Response model for analysis endpoints."""
    success: bool
    session_id: str
    timestamp: str
    processing_time_ms: float
    results: Dict[str, Any]
    feedback: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    models_status: Dict[str, bool]
    system_metrics: Dict[str, Any]


class ABTestConfig(BaseModel):
    """A/B test configuration."""
    test_name: str
    variants: Dict[str, float]  # variant_name -> traffic_percentage
    enabled: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


# Authentication and authorization
class AuthManager:
    """Advanced authentication and authorization manager."""
    
    def __init__(self):
        self.security = HTTPBearer()
        self.audit_manager = AuditTrailManager()
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Validate JWT token and return user info."""
        try:
            # In production, implement proper JWT validation
            token = credentials.credentials
            
            # Mock user validation for demo
            if token == "demo-token":
                user = {
                    "user_id": "demo_user",
                    "role": "medical_professional",
                    "permissions": ["analyze", "view_reports", "manage_sessions"],
                    "organization": "demo_hospital"
                }
                
                # Log authentication event
                self.audit_manager.log_event(
                    event_type=AuditEventType.USER_AUTHENTICATION,
                    description=f"User authenticated: {user['user_id']}",
                    severity=AuditSeverity.INFO,
                    metadata={"user_id": user["user_id"], "role": user["role"]}
                )
                
                return user
            else:
                raise HTTPException(status_code=401, detail="Invalid authentication token")
                
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    def check_permission(self, user: Dict[str, Any], required_permission: str):
        """Check if user has required permission."""
        if required_permission not in user.get("permissions", []):
            raise HTTPException(
                status_code=403, 
                detail=f"Insufficient permissions. Required: {required_permission}"
            )


# A/B Testing framework
class ABTestManager:
    """A/B testing framework for model experiments."""
    
    def __init__(self):
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.redis_client = redis_client
    
    def create_test(self, config: ABTestConfig):
        """Create a new A/B test."""
        self.active_tests[config.test_name] = config
        
        # Store in Redis for persistence
        self.redis_client.hset(
            "ab_tests", 
            config.test_name, 
            json.dumps(config.dict())
        )
    
    def get_variant(self, test_name: str, user_id: str) -> str:
        """Get variant for user based on consistent hashing."""
        if test_name not in self.active_tests:
            return "control"
        
        test_config = self.active_tests[test_name]
        if not test_config.enabled:
            return "control"
        
        # Consistent hashing based on user_id
        hash_value = int(hashlib.md5(f"{test_name}_{user_id}".encode()).hexdigest(), 16)
        hash_percentage = (hash_value % 100) / 100.0
        
        cumulative_percentage = 0.0
        for variant, percentage in test_config.variants.items():
            cumulative_percentage += percentage
            if hash_percentage <= cumulative_percentage:
                return variant
        
        return "control"


# Advanced caching system
class CacheManager:
    """Advanced caching with Redis backend."""
    
    def __init__(self):
        self.redis_client = redis_client
        self.default_ttl = 3600  # 1 hour
    
    def _generate_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_parts = [prefix]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        return ":".join(key_parts)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.default_ttl
            self.redis_client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
            return False


# Initialize managers
auth_manager = AuthManager()
ab_test_manager = ABTestManager()
cache_manager = CacheManager()

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting SMART-TRAIN API v2")
    
    # Initialize models
    global cpr_model, feedback_model
    cpr_model = CPRQualityAssessmentModel()
    cpr_model.load_model()
    
    feedback_model = RealTimeFeedbackModel()
    feedback_model.load_model()
    
    # Setup A/B tests
    ab_test_manager.create_test(ABTestConfig(
        test_name="model_version_test",
        variants={"v1": 0.5, "v2": 0.5},
        enabled=True
    ))
    
    logger.info("SMART-TRAIN API v2 startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SMART-TRAIN API v2")


# Create FastAPI application
app = FastAPI(
    title="SMART-TRAIN API v2",
    description="Enterprise-grade AI-powered medical training platform API",
    version="2.0.0",
    docs_url="/v2/docs",
    redoc_url="/v2/redoc",
    openapi_url="/v2/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Middleware for request/response monitoring
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Monitor all requests for metrics and logging."""
    start_time = time.time()
    
    # Increment active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)
        
        # Collect custom metrics
        metrics_collector.collect_metric(
            "request_count", 1, MetricType.USAGE,
            tags={
                "method": request.method,
                "endpoint": request.url.path,
                "status": str(response.status_code)
            }
        )
        
        metrics_collector.collect_metric(
            "response_time", duration * 1000, MetricType.PERFORMANCE,
            tags={"endpoint": request.url.path}
        )
        
        return response
        
    finally:
        # Decrement active connections
        ACTIVE_CONNECTIONS.dec()


# Health check endpoint
@app.get("/v2/health", response_model=HealthResponse)
async def health_check():
    """Advanced health check with system metrics."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        uptime_seconds=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        models_status={
            "cpr_quality_model": cpr_model.is_loaded if 'cpr_model' in globals() else False,
            "feedback_model": feedback_model.is_loaded if 'feedback_model' in globals() else False
        },
        system_metrics={
            "active_connections": ACTIVE_CONNECTIONS._value._value,
            "total_requests": REQUEST_COUNT._value.sum(),
            "cache_hit_rate": 0.85,  # Mock value
            "memory_usage_mb": 512   # Mock value
        }
    )


# Advanced CPR analysis endpoint with A/B testing
@app.post("/v2/analyze/cpr", response_model=AnalysisResponse)
@limiter.limit("100/minute")
async def analyze_cpr_advanced(
    request: Request,
    analysis_request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(auth_manager.get_current_user)
):
    """
    Advanced CPR analysis with A/B testing and caching.
    """
    try:
        start_time = time.time()
        
        # Check permissions
        auth_manager.check_permission(current_user, "analyze")
        
        # Get A/B test variant
        variant = ab_test_manager.get_variant("model_version_test", current_user["user_id"])
        
        # Generate cache key
        cache_key = cache_manager._generate_cache_key(
            "cpr_analysis",
            data_hash=hashlib.md5(str(analysis_request.data).encode()).hexdigest(),
            variant=variant
        )
        
        # Check cache
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            logger.info("Cache hit for CPR analysis", cache_key=cache_key)
            return AnalysisResponse(**cached_result)
        
        # Perform analysis
        import numpy as np
        pose_data = np.array(analysis_request.data)
        
        # Use appropriate model variant
        if variant == "v2":
            # Use enhanced model (mock for demo)
            cpr_result = cpr_model.predict(pose_data)
        else:
            # Use standard model
            cpr_result = cpr_model.predict(pose_data)
        
        if not cpr_result.success:
            raise HTTPException(status_code=500, detail="CPR analysis failed")
        
        # Generate feedback
        feedback_input = {
            "cpr_metrics": cpr_result.data["cpr_metrics"],
            "overall_score": cpr_result.data["cpr_metrics"]["overall_quality_score"]
        }
        
        feedback_result = feedback_model.predict(feedback_input)
        
        # Create response
        response_data = {
            "success": True,
            "session_id": analysis_request.session_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": (time.time() - start_time) * 1000,
            "results": cpr_result.data,
            "feedback": feedback_result.data if feedback_result.success else None,
            "metadata": {
                "user_id": current_user["user_id"],
                "variant": variant,
                "model_version": cpr_model.model_version,
                "cached": False
            }
        }
        
        # Cache result
        background_tasks.add_task(
            cache_manager.set, 
            cache_key, 
            response_data, 
            ttl=1800  # 30 minutes
        )
        
        # Record model inference
        MODEL_INFERENCE_COUNT.labels(model_name="cpr_quality").inc()
        
        # Log analysis event
        background_tasks.add_task(
            log_analysis_event,
            analysis_request.session_id,
            current_user["user_id"],
            response_data
        )
        
        return AnalysisResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Advanced CPR analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Batch analysis endpoint
@app.post("/v2/analyze/batch")
@limiter.limit("10/minute")
async def analyze_batch(
    request: Request,
    batch_requests: List[AnalysisRequest],
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(auth_manager.get_current_user)
):
    """
    Batch analysis endpoint for processing multiple requests efficiently.
    """
    try:
        auth_manager.check_permission(current_user, "analyze")
        
        if len(batch_requests) > 50:
            raise HTTPException(status_code=400, detail="Batch size too large (max 50)")
        
        # Process requests concurrently
        tasks = []
        for req in batch_requests:
            task = asyncio.create_task(
                process_single_analysis(req, current_user)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from errors
        successful_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    "index": i,
                    "session_id": batch_requests[i].session_id,
                    "error": str(result)
                })
            else:
                successful_results.append(result)
        
        return {
            "success": len(errors) == 0,
            "total_requests": len(batch_requests),
            "successful_results": len(successful_results),
            "errors": len(errors),
            "results": successful_results,
            "errors_detail": errors,
            "processing_time_ms": 0  # Would calculate actual time
        }
        
    except Exception as e:
        logger.error("Batch analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


async def process_single_analysis(analysis_request: AnalysisRequest, user: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single analysis request (helper for batch processing)."""
    import numpy as np
    
    pose_data = np.array(analysis_request.data)
    cpr_result = cpr_model.predict(pose_data)
    
    if not cpr_result.success:
        raise Exception("CPR analysis failed")
    
    return {
        "session_id": analysis_request.session_id,
        "results": cpr_result.data,
        "timestamp": datetime.now().isoformat()
    }


# Metrics endpoint for Prometheus
@app.get("/v2/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


# A/B test management endpoints
@app.post("/v2/admin/ab-tests")
async def create_ab_test(
    test_config: ABTestConfig,
    current_user: dict = Depends(auth_manager.get_current_user)
):
    """Create a new A/B test."""
    auth_manager.check_permission(current_user, "admin")
    
    ab_test_manager.create_test(test_config)
    
    return {
        "success": True,
        "message": f"A/B test '{test_config.test_name}' created successfully"
    }


@app.get("/v2/admin/ab-tests")
async def list_ab_tests(current_user: dict = Depends(auth_manager.get_current_user)):
    """List all active A/B tests."""
    auth_manager.check_permission(current_user, "admin")
    
    return {
        "active_tests": list(ab_test_manager.active_tests.keys()),
        "tests_detail": {
            name: config.dict() 
            for name, config in ab_test_manager.active_tests.items()
        }
    }


# Cache management endpoints
@app.delete("/v2/admin/cache/{cache_key}")
async def invalidate_cache(
    cache_key: str,
    current_user: dict = Depends(auth_manager.get_current_user)
):
    """Invalidate specific cache entry."""
    auth_manager.check_permission(current_user, "admin")
    
    success = await cache_manager.delete(cache_key)
    
    return {
        "success": success,
        "message": f"Cache key '{cache_key}' {'invalidated' if success else 'not found'}"
    }


# Background task for logging
async def log_analysis_event(session_id: str, user_id: str, analysis_result: Dict[str, Any]):
    """Log analysis event to audit trail."""
    audit_manager = AuditTrailManager()
    audit_manager.log_event(
        event_type=AuditEventType.MODEL_INFERENCE,
        description=f"Advanced CPR analysis completed for session {session_id}",
        severity=AuditSeverity.INFO,
        metadata={
            "session_id": session_id,
            "user_id": user_id,
            "processing_time_ms": analysis_result.get("processing_time_ms"),
            "variant": analysis_result.get("metadata", {}).get("variant"),
            "api_version": "2.0.0"
        }
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with detailed logging."""
    logger.error("HTTP exception", 
                status_code=exc.status_code, 
                detail=exc.detail,
                path=request.url.path)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


# Main entry point
if __name__ == "__main__":
    app.state.start_time = time.time()
    
    uvicorn.run(
        "advanced_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
        access_log=True
    )
