"""
Autonomous AI Agents module for SMART-TRAIN platform.

This module implements autonomous AI agents that can:
- Self-improve through continuous learning
- Adapt to individual learner needs
- Collaborate with human instructors
- Make intelligent medical training decisions
- Evolve their teaching strategies
- Provide personalized learning paths
"""

from .autonomous_trainer import AutonomousTrainingAgent, AgentPersonality
from .self_improving_ai import SelfImprovingMedicalAI, ContinuousLearningEngine
from .collaborative_agents import MultiAgentSystem, AgentCollaboration
from .adaptive_intelligence import AdaptiveIntelligenceEngine, PersonalizationAgent
from .decision_making import MedicalDecisionAgent, IntelligentReasoningEngine

__all__ = [
    "AutonomousTrainingAgent",
    "AgentPersonality",
    "SelfImprovingMedicalAI",
    "ContinuousLearningEngine",
    "MultiAgentSystem",
    "AgentCollaboration",
    "AdaptiveIntelligenceEngine",
    "PersonalizationAgent",
    "MedicalDecisionAgent",
    "IntelligentReasoningEngine"
]
