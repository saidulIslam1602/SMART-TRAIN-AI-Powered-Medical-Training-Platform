"""
FastAPI main application for SMART-TRAIN platform.

This module provides the main FastAPI application with real-time
medical training analysis endpoints and WebSocket streaming.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
import cv2
import base64
from datetime import datetime, timezone

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smart_train.core.config import get_config
from smart_train.core.logging import setup_logging, get_logger
from smart_train.models.cpr_quality_model import CPRQualityAssessmentModel
from smart_train.models.realtime_feedback import RealTimeFeedbackModel
from smart_train.data.preprocessing import MedicalDataPreprocessor
from smart_train.compliance.audit_trail import AuditTrailManager, AuditEventType, AuditSeverity
from smart_train.core.exceptions import SmartTrainException

# Initialize configuration and logging
config = get_config()
setup_logging(log_level='INFO')
logger = get_logger(__name__)

# Initialize security
security = HTTPBearer()

# Initialize models (will be loaded on startup)
cpr_quality_model: Optional[CPRQualityAssessmentModel] = None
feedback_model: Optional[RealTimeFeedbackModel] = None
preprocessor: Optional[MedicalDataPreprocessor] = None
audit_manager: Optional[AuditTrailManager] = None

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time analysis."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_info: Dict[str, Any] = None):
        """Accept and store WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            'connected_at': time.time(),
            'client_info': client_info or {},
            'frames_processed': 0,
            'last_activity': time.time()
        }
        logger.info("WebSocket connection established", 
                   total_connections=len(self.active_connections))
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            metadata = self.connection_metadata.pop(websocket, {})
            session_duration = time.time() - metadata.get('connected_at', time.time())
            logger.info("WebSocket connection closed", 
                       session_duration=session_duration,
                       frames_processed=metadata.get('frames_processed', 0))
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific WebSocket."""
        try:
            await websocket.send_text(json.dumps(message))
            self.connection_metadata[websocket]['last_activity'] = time.time()
        except Exception as e:
            logger.error("Failed to send WebSocket message", error=str(e))
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSockets."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

# Initialize connection manager
manager = ConnectionManager()

# Create FastAPI application
app = FastAPI(
    title="SMART-TRAIN AI Platform",
    description="Enterprise-grade AI-powered medical training platform with real-time analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for authentication (simplified for demo)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return user info."""
    # In production, implement proper JWT validation
    return {"user_id": "demo_user", "role": "trainee"}

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    global cpr_quality_model, feedback_model, preprocessor, audit_manager
    
    try:
        logger.info("Starting SMART-TRAIN API server...")
        
        # Initialize audit manager
        audit_manager = AuditTrailManager()
        
        # Initialize models
        cpr_quality_model = CPRQualityAssessmentModel()
        cpr_quality_model.load_model()
        
        feedback_model = RealTimeFeedbackModel()
        feedback_model.load_model()
        
        # Initialize preprocessor
        preprocessor = MedicalDataPreprocessor()
        
        # Log startup event
        audit_manager.log_event(
            event_type=AuditEventType.SYSTEM_STARTUP,
            description="SMART-TRAIN API server started",
            severity=AuditSeverity.INFO,
            metadata={
                'models_loaded': ['CPRQualityAssessment', 'RealTimeFeedback'],
                'api_version': '2.0.0'
            }
        )
        
        logger.info("SMART-TRAIN API server startup complete")
        
    except Exception as e:
        logger.error("Failed to start SMART-TRAIN API server", error=str(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down SMART-TRAIN API server...")
    
    if audit_manager:
        audit_manager.log_event(
            event_type=AuditEventType.SYSTEM_SHUTDOWN,
            description="SMART-TRAIN API server shutdown",
            severity=AuditSeverity.INFO
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "models_loaded": {
            "cpr_quality_model": cpr_quality_model is not None and cpr_quality_model.is_loaded,
            "feedback_model": feedback_model is not None and feedback_model.is_loaded,
            "preprocessor": preprocessor is not None
        }
    }

# Model information endpoint
@app.get("/api/v1/models/info")
async def get_models_info(current_user: dict = Depends(get_current_user)):
    """Get information about loaded models."""
    return {
        "cpr_quality_model": cpr_quality_model.get_model_info() if cpr_quality_model else None,
        "feedback_model": feedback_model.get_model_info() if feedback_model else None,
        "api_version": "2.0.0",
        "capabilities": [
            "real_time_cpr_analysis",
            "quality_assessment",
            "aha_compliance_checking",
            "real_time_feedback",
            "performance_tracking"
        ]
    }

# CPR analysis endpoint
@app.post("/api/v1/analyze/cpr")
async def analyze_cpr(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze CPR performance from pose data.
    
    Expected request format:
    {
        "pose_data": [...],  # Pose landmarks array
        "session_id": "string",
        "metadata": {...}
    }
    """
    try:
        start_time = time.time()
        
        # Validate request
        if "pose_data" not in request:
            raise HTTPException(status_code=400, detail="Missing pose_data in request")
        
        pose_data = np.array(request["pose_data"])
        session_id = request.get("session_id", "unknown")
        
        # CPR quality analysis
        cpr_result = cpr_quality_model.predict(pose_data)
        
        if not cpr_result.success:
            raise HTTPException(status_code=500, detail=f"CPR analysis failed: {cpr_result.error_message}")
        
        # Generate real-time feedback
        feedback_input = {
            "cpr_metrics": cpr_result.data["cpr_metrics"],
            "overall_score": cpr_result.data["cpr_metrics"]["overall_quality_score"]
        }
        
        feedback_result = feedback_model.predict(feedback_input)
        
        # Combine results
        analysis_result = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpr_analysis": cpr_result.data,
            "feedback": feedback_result.data if feedback_result.success else None,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "user_id": current_user["user_id"]
        }
        
        # Log analysis event in background
        background_tasks.add_task(
            log_analysis_event,
            session_id,
            current_user["user_id"],
            analysis_result
        )
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("CPR analysis endpoint failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Real-time WebSocket endpoint
@app.websocket("/api/v1/realtime/analyze")
async def websocket_realtime_analysis(websocket: WebSocket):
    """
    Real-time analysis WebSocket endpoint.
    
    Accepts video frames and returns immediate analysis results.
    """
    await manager.connect(websocket)
    
    try:
        # Send connection confirmation
        await manager.send_personal_message({
            "type": "connection_established",
            "message": "Real-time analysis ready",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, websocket)
        
        frame_count = 0
        session_start = time.time()
        
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            if frame_data.get("type") == "video_frame":
                try:
                    # Decode base64 frame
                    frame_bytes = base64.b64decode(frame_data["frame"])
                    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    # Process frame for pose estimation
                    # (This is a simplified version - in production, implement full pipeline)
                    analysis_start = time.time()
                    
                    # Simulate pose extraction and analysis
                    # In production, use the actual preprocessor and models
                    mock_pose_data = np.random.rand(30, 33, 3)  # Mock pose data
                    
                    # CPR analysis
                    cpr_result = cpr_quality_model.predict(mock_pose_data)
                    
                    if cpr_result.success:
                        # Generate feedback
                        feedback_input = {
                            "cpr_metrics": cpr_result.data["cpr_metrics"],
                            "overall_score": cpr_result.data["cpr_metrics"]["overall_quality_score"]
                        }
                        feedback_result = feedback_model.predict(feedback_input)
                        
                        # Send real-time results
                        response = {
                            "type": "analysis_result",
                            "frame_id": frame_count,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "cpr_metrics": cpr_result.data["cpr_metrics"],
                            "feedback": feedback_result.data["feedback_result"] if feedback_result.success else None,
                            "processing_time_ms": (time.time() - analysis_start) * 1000,
                            "session_duration": time.time() - session_start
                        }
                        
                        await manager.send_personal_message(response, websocket)
                    
                    frame_count += 1
                    manager.connection_metadata[websocket]['frames_processed'] = frame_count
                    
                except Exception as e:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Frame processing error: {str(e)}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }, websocket)
            
            elif frame_data.get("type") == "ping":
                # Respond to ping for connection health
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        manager.disconnect(websocket)

# Performance metrics endpoint
@app.get("/api/v1/metrics/performance")
async def get_performance_metrics(current_user: dict = Depends(get_current_user)):
    """Get system performance metrics."""
    return {
        "cpr_model_stats": cpr_quality_model.get_performance_stats() if cpr_quality_model else None,
        "feedback_model_stats": feedback_model.get_session_summary() if feedback_model else None,
        "websocket_connections": len(manager.active_connections),
        "system_uptime": time.time() - (audit_manager.start_time if audit_manager else time.time()),
        "api_version": "2.0.0"
    }

# Session management endpoint
@app.post("/api/v1/session/start")
async def start_training_session(
    request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Start a new training session."""
    session_id = f"session_{int(time.time())}_{current_user['user_id']}"
    
    session_data = {
        "session_id": session_id,
        "user_id": current_user["user_id"],
        "start_time": datetime.now(timezone.utc).isoformat(),
        "training_type": request.get("training_type", "cpr"),
        "difficulty_level": request.get("difficulty_level", "beginner"),
        "objectives": request.get("objectives", []),
        "status": "active"
    }
    
    # Log session start
    if audit_manager:
        audit_manager.log_event(
            event_type=AuditEventType.TRAINING_SESSION_START,
            description=f"Training session started: {session_id}",
            severity=AuditSeverity.INFO,
            metadata=session_data
        )
    
    return session_data

# Background task for logging
async def log_analysis_event(session_id: str, user_id: str, analysis_result: Dict[str, Any]):
    """Log analysis event to audit trail."""
    if audit_manager:
        audit_manager.log_event(
            event_type=AuditEventType.MODEL_INFERENCE,
            description=f"CPR analysis completed for session {session_id}",
            severity=AuditSeverity.INFO,
            metadata={
                "session_id": session_id,
                "user_id": user_id,
                "quality_score": analysis_result.get("cpr_analysis", {}).get("cpr_metrics", {}).get("overall_quality_score"),
                "processing_time_ms": analysis_result.get("processing_time_ms")
            }
        )

# Error handlers
@app.exception_handler(SmartTrainException)
async def smart_train_exception_handler(request, exc: SmartTrainException):
    """Handle SMART-TRAIN specific exceptions."""
    logger.error("SMART-TRAIN exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": f"SMART-TRAIN error: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.is_development(),
        log_level="info"
    )
