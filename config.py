"""
Configuration file for Text Summarization Model
==============================================

This file contains all configuration settings for the text summarization system.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    model_id: str
    description: str
    max_length: int
    min_length: int
    recommended_for: List[str]

# Available models configuration
AVAILABLE_MODELS = {
    "bart-large-cnn": ModelConfig(
        name="BART Large CNN",
        model_id="facebook/bart-large-cnn",
        description="Best for news articles and general text summarization",
        max_length=200,
        min_length=30,
        recommended_for=["news", "articles", "general"]
    ),
    "bart-base-cnn": ModelConfig(
        name="BART Base CNN",
        model_id="facebook/bart-base-cnn",
        description="Faster alternative with good quality",
        max_length=150,
        min_length=25,
        recommended_for=["news", "articles", "general"]
    ),
    "t5-small": ModelConfig(
        name="T5 Small",
        model_id="t5-small",
        description="Google's Text-to-Text Transfer Transformer (small)",
        max_length=100,
        min_length=20,
        recommended_for=["general", "short_text"]
    ),
    "t5-base": ModelConfig(
        name="T5 Base",
        model_id="t5-base",
        description="Google's Text-to-Text Transfer Transformer (base)",
        max_length=150,
        min_length=30,
        recommended_for=["general", "long_text"]
    ),
    "pegasus-cnn": ModelConfig(
        name="Pegasus CNN",
        model_id="google/pegasus-cnn_dailymail",
        description="Specialized for news summarization",
        max_length=200,
        min_length=30,
        recommended_for=["news", "articles"]
    )
}

# Default configuration
DEFAULT_CONFIG = {
    "model": "bart-large-cnn",
    "max_length": 100,
    "min_length": 30,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "early_stopping": True,
    "batch_size": 1,
    "device": "auto"  # auto, cpu, cuda
}

# UI Configuration
UI_CONFIG = {
    "page_title": "Text Summarization Model",
    "page_icon": "📝",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme": {
        "primary_color": "#667eea",
        "background_color": "#ffffff",
        "secondary_background_color": "#f0f2f6",
        "text_color": "#262730"
    }
}

# Database Configuration
DATABASE_CONFIG = {
    "db_path": "sample_texts.json",
    "backup_path": "sample_texts_backup.json",
    "auto_backup": True,
    "max_articles": 1000
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "summarization.log",
    "max_size": 10485760,  # 10MB
    "backup_count": 5
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "cache_models": True,
    "cache_timeout": 3600,  # 1 hour
    "max_text_length": 10000,
    "min_text_length": 50,
    "enable_gpu": True,
    "mixed_precision": True
}

# Error Messages
ERROR_MESSAGES = {
    "model_load_error": "Failed to load the selected model. Please try a different model or check your internet connection.",
    "text_too_short": "Text must be at least 50 characters long for summarization.",
    "text_too_long": "Text is too long. Please limit to 10,000 characters.",
    "summarization_error": "An error occurred during summarization. Please try again.",
    "file_upload_error": "Error uploading file. Please ensure it's a valid text file.",
    "database_error": "Database operation failed. Please try again."
}

# Success Messages
SUCCESS_MESSAGES = {
    "model_loaded": "Model loaded successfully!",
    "summary_generated": "Summary generated successfully!",
    "file_uploaded": "File uploaded successfully!",
    "article_saved": "Article saved successfully!",
    "settings_saved": "Settings saved successfully!"
}

# File Upload Configuration
FILE_UPLOAD_CONFIG = {
    "allowed_types": ["txt", "md", "docx"],
    "max_size_mb": 10,
    "encoding": "utf-8"
}

# Export Configuration
EXPORT_CONFIG = {
    "formats": ["txt", "json", "csv"],
    "include_metadata": True,
    "timestamp_format": "%Y-%m-%d_%H-%M-%S"
}

def get_model_config(model_key: str) -> ModelConfig:
    """Get configuration for a specific model."""
    return AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["bart-large-cnn"])

def get_default_model_config() -> ModelConfig:
    """Get the default model configuration."""
    return get_model_config(DEFAULT_CONFIG["model"])

def validate_config(config: Dict) -> bool:
    """Validate configuration parameters."""
    required_keys = ["model", "max_length", "min_length"]
    return all(key in config for key in required_keys)

def get_environment_config() -> Dict:
    """Get configuration from environment variables."""
    env_config = {}
    
    # Model configuration
    if os.getenv("SUMMARIZATION_MODEL"):
        env_config["model"] = os.getenv("SUMMARIZATION_MODEL")
    
    # Performance configuration
    if os.getenv("ENABLE_GPU"):
        env_config["enable_gpu"] = os.getenv("ENABLE_GPU").lower() == "true"
    
    if os.getenv("MAX_TEXT_LENGTH"):
        try:
            env_config["max_text_length"] = int(os.getenv("MAX_TEXT_LENGTH"))
        except ValueError:
            pass
    
    return env_config
