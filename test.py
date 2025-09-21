"""
Test script for Text Summarization Model
========================================

Simple test script to verify the functionality of the text summarization system.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import AVAILABLE_MODELS, DEFAULT_CONFIG, get_model_config
        print(f"✅ Configuration loaded successfully")
        print(f"   Available models: {len(AVAILABLE_MODELS)}")
        print(f"   Default model: {DEFAULT_CONFIG['model']}")
        
        # Test model config retrieval
        model_config = get_model_config("bart-large-cnn")
        print(f"   Model config: {model_config.name}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_database():
    """Test mock database functionality."""
    print("\n🧪 Testing mock database...")
    
    try:
        from 0104 import MockDatabase
        
        # Initialize database
        db = MockDatabase()
        print("✅ Database initialized")
        
        # Test getting articles
        articles = db.get_all_articles()
        print(f"✅ Retrieved {len(articles)} articles")
        
        if articles:
            # Test getting article by ID
            article = db.get_article_by_id(1)
            if article:
                print(f"✅ Retrieved article by ID: {article['title']}")
            else:
                print("❌ Failed to retrieve article by ID")
                return False
            
            # Test getting articles by category
            tech_articles = db.get_articles_by_category("Technology")
            print(f"✅ Retrieved {len(tech_articles)} technology articles")
        
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_summarizer():
    """Test text summarizer functionality."""
    print("\n🧪 Testing text summarizer...")
    
    try:
        from 0104 import TextSummarizer, SummarizationConfig
        
        # Test configuration
        config = SummarizationConfig(
            model_name="facebook/bart-base-cnn",  # Use smaller model for testing
            max_length=50,
            min_length=20
        )
        print("✅ Configuration created")
        
        # Test summarizer initialization (this might take time)
        print("   Loading model (this may take a moment)...")
        summarizer = TextSummarizer(config)
        print("✅ Summarizer initialized")
        
        # Test summarization
        test_text = """
        Artificial Intelligence (AI) is rapidly transforming our world. 
        From self-driving cars and personalized recommendations to virtual assistants and automated customer service, 
        AI applications are everywhere. The technology uses algorithms and large datasets to learn patterns, 
        make decisions, and even generate human-like text and images.
        """
        
        result = summarizer.summarize(test_text)
        print("✅ Summarization completed")
        print(f"   Original length: {result['original_length']} characters")
        print(f"   Summary length: {result['summary_length']} characters")
        print(f"   Summary: {result['summary'][:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Summarizer test failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability."""
    print("\n🧪 Testing CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("ℹ️  CUDA not available, using CPU")
        return True
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Text Summarization Model - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Database", test_database),
        ("CUDA", test_cuda),
        ("Summarizer", test_summarizer),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n🚀 To run the web interface:")
        print("   streamlit run app.py")
        print("\n🚀 To run the command line version:")
        print("   python 0104.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("   You may need to install missing dependencies or check your configuration.")

if __name__ == "__main__":
    main()
