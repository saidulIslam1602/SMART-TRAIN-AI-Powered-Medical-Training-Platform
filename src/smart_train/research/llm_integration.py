"""
Large Language Model Integration for Intelligent Medical Feedback.

This module integrates state-of-the-art LLMs to provide:
- Intelligent, contextual medical feedback
- Natural language explanations of technique issues
- Personalized learning recommendations
- Medical knowledge integration
- Multi-language support for global deployment
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, BitsAndBytesConfig
)
import torch
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import tiktoken

from ..core.base import BaseModel, ProcessingResult
from ..core.logging import get_logger
from ..core.exceptions import ModelTrainingError
from ..compliance.audit_trail import AuditTrailManager, AuditEventType, AuditSeverity

logger = get_logger(__name__)


class FeedbackType(Enum):
    """Types of medical feedback."""
    TECHNIQUE_CORRECTION = "technique_correction"
    ENCOURAGEMENT = "encouragement"
    SAFETY_WARNING = "safety_warning"
    PROGRESS_UPDATE = "progress_update"
    KNOWLEDGE_EXPLANATION = "knowledge_explanation"
    PERSONALIZED_RECOMMENDATION = "personalized_recommendation"


class LanguageCode(Enum):
    """Supported languages for international deployment."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    MANDARIN = "zh"
    JAPANESE = "ja"
    ARABIC = "ar"


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    # Model configuration
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Medical context
    medical_specialty: str = "emergency_medicine"
    certification_level: str = "basic_life_support"
    target_audience: str = "healthcare_professionals"
    
    # Personalization
    enable_personalization: bool = True
    learning_style_adaptation: bool = True
    cultural_sensitivity: bool = True
    
    # Safety and compliance
    medical_accuracy_check: bool = True
    content_filtering: bool = True
    audit_all_responses: bool = True
    
    # Performance
    response_timeout: int = 10
    max_retries: int = 3
    enable_caching: bool = True


@dataclass
class MedicalContext:
    """Medical context for LLM prompts."""
    procedure_type: str
    patient_demographics: Dict[str, Any]
    training_level: str
    session_history: List[Dict[str, Any]]
    current_metrics: Dict[str, float]
    improvement_areas: List[str]
    safety_concerns: List[str]
    cultural_context: Optional[str] = None
    language_preference: LanguageCode = LanguageCode.ENGLISH


class MedicalPromptTemplates:
    """Medical-specific prompt templates for different feedback types."""
    
    TECHNIQUE_CORRECTION = """
    You are an expert medical instructor providing feedback on CPR technique.
    
    Current Performance Metrics:
    - Compression Depth: {compression_depth:.1%}
    - Compression Rate: {compression_rate:.1%}
    - Hand Position: {hand_position:.1%}
    - Release Quality: {release_quality:.1%}
    - Overall Score: {overall_score:.1%}
    
    Areas Needing Improvement: {improvement_areas}
    Safety Concerns: {safety_concerns}
    
    Training Context:
    - Trainee Level: {training_level}
    - Procedure: {procedure_type}
    - Session Number: {session_number}
    
    Provide specific, actionable feedback to help improve technique. Focus on the most critical issues first.
    Be encouraging but direct about safety concerns. Use medical terminology appropriate for the trainee level.
    
    Language: {language}
    Cultural Context: {cultural_context}
    
    Feedback:
    """
    
    SAFETY_WARNING = """
    You are a medical safety expert. A trainee is performing CPR with concerning technique issues.
    
    CRITICAL SAFETY CONCERNS:
    {safety_concerns}
    
    Current Metrics:
    {current_metrics}
    
    Trainee Context:
    - Level: {training_level}
    - Experience: {experience_level}
    
    Provide an IMMEDIATE, clear safety warning. Explain why these issues are dangerous and what 
    could happen to a real patient. Give specific corrective actions.
    
    Use appropriate urgency level while remaining professional and educational.
    
    Language: {language}
    
    Safety Warning:
    """
    
    PERSONALIZED_RECOMMENDATION = """
    You are an adaptive medical education AI creating personalized learning recommendations.
    
    Trainee Profile:
    - Learning Style: {learning_style}
    - Strengths: {strengths}
    - Improvement Areas: {improvement_areas}
    - Progress Trend: {progress_trend}
    - Session History: {session_history}
    
    Performance Analysis:
    {performance_analysis}
    
    Cultural Background: {cultural_context}
    Language: {language}
    
    Create a personalized learning plan with:
    1. Specific practice exercises
    2. Learning resources
    3. Timeline for improvement
    4. Motivation strategies
    
    Recommendation:
    """
    
    KNOWLEDGE_EXPLANATION = """
    You are a medical educator explaining CPR concepts to healthcare professionals.
    
    Topic to Explain: {topic}
    Current Understanding Level: {understanding_level}
    Specific Question: {question}
    
    Context:
    - Trainee Level: {training_level}
    - Medical Background: {medical_background}
    - Learning Objective: {learning_objective}
    
    Provide a clear, accurate explanation that:
    1. Explains the medical rationale
    2. Connects to current guidelines (AHA 2020)
    3. Includes practical application
    4. Uses appropriate medical terminology
    
    Language: {language}
    
    Explanation:
    """


class MedicalLLMAgent:
    """
    Advanced LLM agent specialized for medical training feedback.
    
    This agent provides intelligent, contextual feedback using state-of-the-art
    language models with medical knowledge integration.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.audit_manager = AuditTrailManager()
        
        # Initialize LLM based on configuration
        self._initialize_llm()
        
        # Medical knowledge base
        self.medical_knowledge = self._load_medical_knowledge()
        
        # Conversation memory for personalization
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Response cache for performance
        self.response_cache = {} if config.enable_caching else None
        
        logger.info("Medical LLM Agent initialized", 
                   model=config.model_name,
                   medical_specialty=config.medical_specialty)
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on configuration."""
        try:
            if self.config.model_name.startswith("gpt"):
                # OpenAI GPT models
                self.llm = OpenAI(
                    model_name=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    request_timeout=self.config.response_timeout
                )
                
            elif "llama" in self.config.model_name.lower():
                # Local Llama models
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                
                # Quantization for efficiency
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                
                self.llm = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True
                )
            
            else:
                raise ValueError(f"Unsupported model: {self.config.model_name}")
                
        except Exception as e:
            logger.error("Failed to initialize LLM", error=str(e))
            raise ModelTrainingError(f"LLM initialization failed: {e}")
    
    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Load medical knowledge base for context enhancement."""
        # In production, this would load from a comprehensive medical database
        return {
            "cpr_guidelines": {
                "aha_2020": {
                    "compression_depth": "5-6 cm (2-2.4 inches)",
                    "compression_rate": "100-120 per minute",
                    "compression_fraction": ">60%",
                    "ventilation_ratio": "30:2 for adults"
                }
            },
            "safety_protocols": {
                "scene_safety": "Ensure scene is safe before approaching",
                "infection_control": "Use appropriate PPE",
                "team_communication": "Clear, closed-loop communication"
            },
            "common_errors": {
                "insufficient_depth": "Most common error - leads to ineffective circulation",
                "excessive_rate": "Can cause fatigue and reduced effectiveness",
                "incomplete_recoil": "Prevents venous return and cardiac filling"
            }
        }
    
    async def generate_feedback(self, feedback_type: FeedbackType, 
                              medical_context: MedicalContext,
                              additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate intelligent medical feedback using LLM.
        
        Args:
            feedback_type: Type of feedback to generate
            medical_context: Medical context and performance data
            additional_context: Additional context information
        
        Returns:
            Dictionary containing generated feedback and metadata
        """
        try:
            # Create cache key
            cache_key = self._create_cache_key(feedback_type, medical_context)
            
            # Check cache first
            if self.response_cache and cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                logger.info("Using cached LLM response", cache_key=cache_key[:50])
                return cached_response
            
            # Select appropriate prompt template
            prompt_template = self._get_prompt_template(feedback_type)
            
            # Prepare prompt variables
            prompt_vars = self._prepare_prompt_variables(medical_context, additional_context)
            
            # Generate response with retries
            response = await self._generate_with_retries(prompt_template, prompt_vars)
            
            # Post-process and validate response
            processed_response = self._process_response(response, feedback_type, medical_context)
            
            # Cache response
            if self.response_cache:
                self.response_cache[cache_key] = processed_response
            
            # Audit logging
            if self.config.audit_all_responses:
                self.audit_manager.log_event(
                    event_type=AuditEventType.MODEL_INFERENCE,
                    description=f"LLM feedback generated: {feedback_type.value}",
                    severity=AuditSeverity.INFO,
                    metadata={
                        "feedback_type": feedback_type.value,
                        "language": medical_context.language_preference.value,
                        "training_level": medical_context.training_level,
                        "response_length": len(processed_response.get("feedback", ""))
                    }
                )
            
            return processed_response
            
        except Exception as e:
            logger.error("LLM feedback generation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "fallback_feedback": self._get_fallback_feedback(feedback_type, medical_context)
            }
    
    def _get_prompt_template(self, feedback_type: FeedbackType) -> str:
        """Get appropriate prompt template for feedback type."""
        template_map = {
            FeedbackType.TECHNIQUE_CORRECTION: MedicalPromptTemplates.TECHNIQUE_CORRECTION,
            FeedbackType.SAFETY_WARNING: MedicalPromptTemplates.SAFETY_WARNING,
            FeedbackType.PERSONALIZED_RECOMMENDATION: MedicalPromptTemplates.PERSONALIZED_RECOMMENDATION,
            FeedbackType.KNOWLEDGE_EXPLANATION: MedicalPromptTemplates.KNOWLEDGE_EXPLANATION,
        }
        
        return template_map.get(feedback_type, MedicalPromptTemplates.TECHNIQUE_CORRECTION)
    
    def _prepare_prompt_variables(self, medical_context: MedicalContext, 
                                additional_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare variables for prompt template."""
        base_vars = {
            "compression_depth": medical_context.current_metrics.get("compression_depth", 0),
            "compression_rate": medical_context.current_metrics.get("compression_rate", 0),
            "hand_position": medical_context.current_metrics.get("hand_position", 0),
            "release_quality": medical_context.current_metrics.get("release_quality", 0),
            "overall_score": medical_context.current_metrics.get("overall_score", 0),
            "improvement_areas": ", ".join(medical_context.improvement_areas),
            "safety_concerns": ", ".join(medical_context.safety_concerns),
            "training_level": medical_context.training_level,
            "procedure_type": medical_context.procedure_type,
            "session_number": len(medical_context.session_history) + 1,
            "language": medical_context.language_preference.value,
            "cultural_context": medical_context.cultural_context or "General"
        }
        
        # Add additional context
        if additional_context:
            base_vars.update(additional_context)
        
        return base_vars
    
    async def _generate_with_retries(self, prompt_template: str, 
                                   prompt_vars: Dict[str, Any]) -> str:
        """Generate LLM response with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                # Format prompt
                formatted_prompt = prompt_template.format(**prompt_vars)
                
                # Generate response
                if hasattr(self.llm, 'agenerate'):
                    # Async generation
                    response = await self.llm.agenerate([formatted_prompt])
                    return response.generations[0][0].text
                else:
                    # Sync generation
                    if isinstance(self.llm, pipeline):
                        # Hugging Face pipeline
                        result = self.llm(formatted_prompt, max_new_tokens=self.config.max_tokens)
                        return result[0]['generated_text'][len(formatted_prompt):]
                    else:
                        # LangChain LLM
                        return self.llm(formatted_prompt)
                        
            except Exception as e:
                logger.warning(f"LLM generation attempt {attempt + 1} failed", error=str(e))
                if attempt == self.config.max_retries - 1:
                    raise
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)
        
        raise Exception("All LLM generation attempts failed")
    
    def _process_response(self, response: str, feedback_type: FeedbackType, 
                         medical_context: MedicalContext) -> Dict[str, Any]:
        """Post-process and validate LLM response."""
        # Clean up response
        cleaned_response = response.strip()
        
        # Medical accuracy check
        if self.config.medical_accuracy_check:
            accuracy_score = self._check_medical_accuracy(cleaned_response)
        else:
            accuracy_score = 1.0
        
        # Content filtering
        if self.config.content_filtering:
            is_appropriate = self._check_content_appropriateness(cleaned_response)
        else:
            is_appropriate = True
        
        # Extract structured information
        structured_info = self._extract_structured_info(cleaned_response, feedback_type)
        
        return {
            "success": True,
            "feedback": cleaned_response,
            "feedback_type": feedback_type.value,
            "language": medical_context.language_preference.value,
            "structured_info": structured_info,
            "quality_metrics": {
                "medical_accuracy": accuracy_score,
                "content_appropriate": is_appropriate,
                "response_length": len(cleaned_response),
                "readability_score": self._calculate_readability(cleaned_response)
            },
            "metadata": {
                "generation_timestamp": time.time(),
                "model_used": self.config.model_name,
                "training_level": medical_context.training_level
            }
        }
    
    def _check_medical_accuracy(self, response: str) -> float:
        """Check medical accuracy of response against knowledge base."""
        # Simplified accuracy check - in production would use more sophisticated methods
        accuracy_keywords = [
            "compression", "depth", "rate", "recoil", "position",
            "airway", "breathing", "circulation", "AHA", "guidelines"
        ]
        
        found_keywords = sum(1 for keyword in accuracy_keywords if keyword.lower() in response.lower())
        return min(found_keywords / len(accuracy_keywords), 1.0)
    
    def _check_content_appropriateness(self, response: str) -> bool:
        """Check if content is appropriate for medical training."""
        # Basic content filtering - would be more sophisticated in production
        inappropriate_terms = ["harmful", "dangerous", "ignore", "skip"]
        return not any(term in response.lower() for term in inappropriate_terms)
    
    def _extract_structured_info(self, response: str, feedback_type: FeedbackType) -> Dict[str, Any]:
        """Extract structured information from response."""
        # Simplified extraction - would use NLP techniques in production
        structured = {
            "key_points": [],
            "action_items": [],
            "safety_notes": [],
            "references": []
        }
        
        # Extract bullet points and numbered lists
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*')) or line[0:2].isdigit():
                structured["key_points"].append(line)
            elif "safety" in line.lower() or "warning" in line.lower():
                structured["safety_notes"].append(line)
        
        return structured
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score for the response."""
        # Simplified readability calculation
        words = len(text.split())
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        if sentences == 0:
            return 0.5
        
        avg_words_per_sentence = words / sentences
        
        # Flesch-like score (simplified)
        readability = max(0, min(1, (20 - avg_words_per_sentence) / 20))
        return readability
    
    def _create_cache_key(self, feedback_type: FeedbackType, 
                         medical_context: MedicalContext) -> str:
        """Create cache key for response caching."""
        key_components = [
            feedback_type.value,
            medical_context.procedure_type,
            medical_context.training_level,
            str(hash(tuple(sorted(medical_context.current_metrics.items())))),
            medical_context.language_preference.value
        ]
        return "_".join(key_components)
    
    def _get_fallback_feedback(self, feedback_type: FeedbackType, 
                             medical_context: MedicalContext) -> str:
        """Generate fallback feedback when LLM fails."""
        fallback_messages = {
            FeedbackType.TECHNIQUE_CORRECTION: "Please focus on maintaining proper compression depth and rate according to AHA guidelines.",
            FeedbackType.SAFETY_WARNING: "Please review your technique carefully and ensure patient safety protocols are followed.",
            FeedbackType.ENCOURAGEMENT: "Keep practicing! Your technique is improving with each session.",
            FeedbackType.PROGRESS_UPDATE: "Continue working on the areas identified for improvement."
        }
        
        return fallback_messages.get(feedback_type, "Please continue practicing and focus on proper technique.")


class IntelligentFeedbackGenerator(BaseModel):
    """
    Complete intelligent feedback generation system.
    
    This system combines LLM capabilities with medical knowledge
    to provide comprehensive, personalized feedback for medical training.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__("IntelligentFeedbackGenerator", "4.0.0")
        
        self.config = config or LLMConfig()
        self.llm_agent = MedicalLLMAgent(self.config)
        
        # Feedback history for personalization
        self.feedback_history = {}
        
        # Performance tracking
        self.generation_metrics = {
            "total_requests": 0,
            "successful_generations": 0,
            "average_response_time": 0,
            "cache_hit_rate": 0
        }
        
        self.is_loaded = True
        logger.info("Intelligent Feedback Generator initialized")
    
    def load_model(self, model_path: Optional[str] = None):
        """Load model configuration and history."""
        self.is_loaded = True
    
    async def predict(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """
        Generate intelligent medical feedback.
        
        Args:
            input_data: Dictionary containing:
                - feedback_type: Type of feedback needed
                - medical_context: Medical context and performance data
                - user_id: User identifier for personalization
                - additional_context: Optional additional context
        
        Returns:
            ProcessingResult with generated feedback
        """
        try:
            start_time = time.time()
            self.inference_count += 1
            self.generation_metrics["total_requests"] += 1
            
            # Extract input parameters
            feedback_type_str = input_data.get("feedback_type", "technique_correction")
            feedback_type = FeedbackType(feedback_type_str)
            
            medical_context_data = input_data.get("medical_context", {})
            medical_context = MedicalContext(**medical_context_data)
            
            user_id = input_data.get("user_id")
            additional_context = input_data.get("additional_context", {})
            
            # Add personalization context
            if user_id and user_id in self.feedback_history:
                additional_context["user_history"] = self.feedback_history[user_id]
            
            # Generate feedback
            feedback_result = await self.llm_agent.generate_feedback(
                feedback_type, medical_context, additional_context
            )
            
            # Update feedback history
            if user_id and feedback_result.get("success"):
                if user_id not in self.feedback_history:
                    self.feedback_history[user_id] = []
                
                self.feedback_history[user_id].append({
                    "timestamp": time.time(),
                    "feedback_type": feedback_type.value,
                    "response": feedback_result["feedback"][:100] + "...",  # Truncated for storage
                    "quality_score": feedback_result["quality_metrics"]["medical_accuracy"]
                })
                
                # Keep only recent history
                if len(self.feedback_history[user_id]) > 10:
                    self.feedback_history[user_id] = self.feedback_history[user_id][-10:]
            
            # Update metrics
            processing_time = time.time() - start_time
            if feedback_result.get("success"):
                self.generation_metrics["successful_generations"] += 1
            
            # Update average response time
            current_avg = self.generation_metrics["average_response_time"]
            total_requests = self.generation_metrics["total_requests"]
            self.generation_metrics["average_response_time"] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
            
            # Prepare results
            results = {
                "feedback_generation": feedback_result,
                "personalization": {
                    "user_id": user_id,
                    "feedback_history_length": len(self.feedback_history.get(user_id, [])),
                    "personalized": user_id is not None
                },
                "performance_metrics": {
                    "generation_time_ms": processing_time * 1000,
                    "model_version": self.model_version,
                    "inference_count": self.inference_count
                }
            }
            
            return ProcessingResult(
                success=feedback_result.get("success", False),
                data=results,
                metadata={
                    "processing_time_ms": processing_time * 1000,
                    "feedback_type": feedback_type.value,
                    "language": medical_context.language_preference.value
                }
            )
            
        except Exception as e:
            logger.error("Intelligent feedback generation failed", error=str(e))
            return ProcessingResult(
                success=False,
                error_message=f"Feedback generation failed: {str(e)}",
                data={}
            )
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get feedback generation statistics."""
        return {
            "performance_metrics": self.generation_metrics,
            "user_statistics": {
                "total_users": len(self.feedback_history),
                "average_feedback_per_user": (
                    sum(len(history) for history in self.feedback_history.values()) / 
                    len(self.feedback_history) if self.feedback_history else 0
                )
            },
            "model_info": {
                "model_version": self.model_version,
                "total_inferences": self.inference_count,
                "llm_model": self.config.model_name
            }
        }
