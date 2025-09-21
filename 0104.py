"""
Project 104: Advanced Text Summarization Model
==============================================

A modern text summarization system using state-of-the-art transformer models.
Supports multiple models (T5, BART, Pegasus) with configurable parameters.

Features:
- Multiple pre-trained models
- Configurable summarization parameters
- Error handling and validation
- Mock database for testing
- Modern web UI with Streamlit
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    BartForConditionalGeneration
)
import streamlit as st
from streamlit import components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SummarizationConfig:
    """Configuration class for summarization parameters."""
    model_name: str = "facebook/bart-large-cnn"
    max_length: int = 150
    min_length: int = 30
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = True

class TextSummarizer:
    """Advanced text summarization class with multiple model support."""
    
    def __init__(self, config: SummarizationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the specified model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def summarize(self, text: str) -> Dict[str, str]:
        """
        Summarize the input text.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Dictionary containing original text and summary
        """
        try:
            # Validate input
            if not text or len(text.strip()) < 50:
                raise ValueError("Text must be at least 50 characters long")
            
            # Clean and prepare text
            text = text.strip()
            
            # Generate summary
            summary_result = self.pipeline(
                text,
                max_length=self.config.max_length,
                min_length=self.config.min_length,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                length_penalty=self.config.length_penalty,
                early_stopping=self.config.early_stopping
            )
            
            return {
                "original_text": text,
                "summary": summary_result[0]['summary_text'],
                "model_used": self.config.model_name,
                "original_length": len(text),
                "summary_length": len(summary_result[0]['summary_text'])
            }
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            raise
    
    def batch_summarize(self, texts: List[str]) -> List[Dict[str, str]]:
        """Summarize multiple texts in batch."""
        results = []
        for text in texts:
            try:
                result = self.summarize(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error summarizing text: {e}")
                results.append({
                    "original_text": text,
                    "summary": f"Error: {str(e)}",
                    "model_used": self.config.model_name,
                    "original_length": len(text),
                    "summary_length": 0
                })
        return results

class MockDatabase:
    """Mock database for storing and retrieving sample texts."""
    
    def __init__(self, db_path: str = "sample_texts.json"):
        self.db_path = Path(db_path)
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, List[Dict]]:
        """Load sample texts from JSON file."""
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create default sample texts
            default_data = {
                "articles": [
                    {
                        "id": 1,
                        "title": "Artificial Intelligence Revolution",
                        "content": """
                        Artificial Intelligence (AI) is rapidly transforming our world. 
                        From self-driving cars and personalized recommendations to virtual assistants and automated customer service, 
                        AI applications are everywhere. The technology uses algorithms and large datasets to learn patterns, 
                        make decisions, and even generate human-like text and images. While AI offers immense potential, 
                        it also raises ethical concerns around bias, job displacement, and privacy. 
                        Understanding how AI works is becoming increasingly important for both individuals and organizations.
                        """,
                        "category": "Technology"
                    },
                    {
                        "id": 2,
                        "title": "Climate Change and Renewable Energy",
                        "content": """
                        Climate change represents one of the most pressing challenges of our time. 
                        Rising global temperatures, melting ice caps, and extreme weather events are clear indicators 
                        of the urgent need for action. Renewable energy sources like solar, wind, and hydroelectric power 
                        offer promising solutions to reduce greenhouse gas emissions. Governments worldwide are implementing 
                        policies to accelerate the transition to clean energy, while businesses are investing heavily 
                        in sustainable technologies. The shift towards renewable energy not only helps combat climate change 
                        but also creates new economic opportunities and jobs in the green energy sector.
                        """,
                        "category": "Environment"
                    },
                    {
                        "id": 3,
                        "title": "The Future of Remote Work",
                        "content": """
                        The COVID-19 pandemic fundamentally changed how we work, accelerating the adoption of remote work 
                        technologies and practices. Companies that previously resisted remote work were forced to adapt quickly, 
                        implementing video conferencing, cloud-based collaboration tools, and flexible work arrangements. 
                        Studies show that remote work can increase productivity and employee satisfaction while reducing 
                        overhead costs for businesses. However, challenges remain, including maintaining company culture, 
                        ensuring effective communication, and addressing the digital divide. As we move forward, hybrid work 
                        models that combine remote and in-office work are becoming the new standard.
                        """,
                        "category": "Business"
                    }
                ]
            }
            self._save_data(default_data)
            return default_data
    
    def _save_data(self, data: Dict):
        """Save data to JSON file."""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_all_articles(self) -> List[Dict]:
        """Get all articles from the database."""
        return self.data.get("articles", [])
    
    def get_article_by_id(self, article_id: int) -> Optional[Dict]:
        """Get a specific article by ID."""
        articles = self.data.get("articles", [])
        for article in articles:
            if article["id"] == article_id:
                return article
        return None
    
    def get_articles_by_category(self, category: str) -> List[Dict]:
        """Get articles by category."""
        articles = self.data.get("articles", [])
        return [article for article in articles if article["category"].lower() == category.lower()]
    
    def add_article(self, title: str, content: str, category: str) -> int:
        """Add a new article to the database."""
        articles = self.data.get("articles", [])
        new_id = max([a["id"] for a in articles], default=0) + 1
        
        new_article = {
            "id": new_id,
            "title": title,
            "content": content.strip(),
            "category": category
        }
        
        articles.append(new_article)
        self.data["articles"] = articles
        self._save_data(self.data)
        return new_id

def main():
    """Main function to demonstrate the text summarization system."""
    print("🚀 Advanced Text Summarization System")
    print("=" * 50)
    
    # Initialize configuration
    config = SummarizationConfig(
        model_name="facebook/bart-large-cnn",
        max_length=100,
        min_length=30
    )
    
    # Initialize summarizer
    try:
        summarizer = TextSummarizer(config)
        print(f"✅ Model loaded: {config.model_name}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize mock database
    db = MockDatabase()
    articles = db.get_all_articles()
    
    print(f"\n📚 Found {len(articles)} sample articles in database")
    
    # Demonstrate summarization
    for article in articles[:2]:  # Process first 2 articles
        print(f"\n📖 Article: {article['title']}")
        print(f"📊 Category: {article['category']}")
        print(f"📏 Original length: {len(article['content'])} characters")
        
        try:
            result = summarizer.summarize(article['content'])
            print(f"📝 Summary ({result['summary_length']} chars):")
            print(f"   {result['summary']}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()