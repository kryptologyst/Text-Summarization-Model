# Advanced Text Summarization Model

A modern, comprehensive text summarization system using state-of-the-art transformer models. This project demonstrates advanced NLP techniques with a user-friendly web interface and robust backend architecture.

## Features

### Core Functionality
- **Multiple Model Support**: BART, T5, and Pegasus models
- **Configurable Parameters**: Adjustable summary length, temperature, and sampling
- **Interactive Web UI**: Modern Streamlit-based interface
- **Mock Database**: Built-in sample articles for testing
- **Error Handling**: Comprehensive validation and error management
- **Performance Metrics**: Processing time and compression ratio tracking

### Technical Features
- **GPU Acceleration**: Automatic CUDA support when available
- **Model Caching**: Improved performance with cached models
- **Batch Processing**: Support for multiple text summarization
- **File Upload**: Support for text file uploads
- **Export Functionality**: Download summaries in multiple formats
- **Responsive Design**: Mobile-friendly interface

## Requirements

### System Requirements
- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU with CUDA support (optional but recommended)

### Dependencies
- PyTorch 2.0+
- Transformers 4.30+
- Streamlit 1.25+
- Additional packages listed in `requirements.txt`

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd 0104_Text_summarization_model
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

## Quick Start

### Command Line Interface
```bash
# Run the main script
python 0104.py

# This will:
# 1. Load the BART Large CNN model
# 2. Process sample articles from the mock database
# 3. Display summaries with metrics
```

### Web Interface
```bash
# Launch the Streamlit app
streamlit run app.py

# The app will open in your browser at http://localhost:8501
```

## Usage Guide

### Web Interface

1. **Launch the App**: Run `streamlit run app.py`
2. **Select Model**: Choose from available models in the sidebar
3. **Configure Parameters**: Adjust summary length and other settings
4. **Input Text**: Either type/paste text or upload a file
5. **Generate Summary**: Click the "Generate Summary" button
6. **View Results**: Review the summary and metrics
7. **Download**: Save the summary if needed

### Programmatic Usage

```python
from 0104 import TextSummarizer, SummarizationConfig

# Configure the summarizer
config = SummarizationConfig(
    model_name="facebook/bart-large-cnn",
    max_length=100,
    min_length=30
)

# Initialize summarizer
summarizer = TextSummarizer(config)

# Summarize text
text = "Your long text here..."
result = summarizer.summarize(text)

print(f"Summary: {result['summary']}")
print(f"Compression ratio: {result['summary_length']/result['original_length']*100:.1f}%")
```

### Mock Database Usage

```python
from 0104 import MockDatabase

# Initialize database
db = MockDatabase()

# Get all articles
articles = db.get_all_articles()

# Get article by ID
article = db.get_article_by_id(1)

# Get articles by category
tech_articles = db.get_articles_by_category("Technology")

# Add new article
new_id = db.add_article(
    title="New Article",
    content="Article content...",
    category="Technology"
)
```

## 🔧 Configuration

### Model Configuration
The system supports multiple models with different characteristics:

| Model | Best For | Speed | Quality |
|-------|----------|-------|---------|
| BART Large CNN | News, Articles | Medium | High |
| BART Base CNN | General Text | Fast | Good |
| T5 Small | Short Text | Very Fast | Good |
| T5 Base | General Text | Medium | High |
| Pegasus CNN | News Articles | Medium | High |

### Parameter Tuning

- **max_length**: Maximum tokens in summary (20-200)
- **min_length**: Minimum tokens in summary (10-100)
- **do_sample**: Enable sampling for diversity
- **temperature**: Controls randomness (0.1-2.0)
- **repetition_penalty**: Reduces repetition (1.0-2.0)

## Performance

### Benchmarks
- **BART Large CNN**: ~2-5 seconds per article (GPU)
- **T5 Small**: ~1-2 seconds per article (GPU)
- **Memory Usage**: 2-8GB depending on model
- **Text Length**: Supports up to 10,000 characters

### Optimization Tips
1. Use GPU when available for faster processing
2. Enable model caching for repeated use
3. Adjust batch size based on available memory
4. Use smaller models for faster processing

## Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/
```

### Test Coverage
- Model loading and initialization
- Text summarization functionality
- Error handling and validation
- Database operations
- Web interface components

## 📁 Project Structure

```
0104_Text_summarization_model/
├── 0104.py              # Main application code
├── app.py               # Streamlit web interface
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
├── README.md           # This file
├── sample_texts.json   # Mock database (generated)
└── tests/              # Test files (optional)
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check internet connection
   - Verify model name spelling
   - Ensure sufficient disk space

2. **Memory Issues**
   - Use smaller models (T5-small)
   - Reduce batch size
   - Close other applications

3. **CUDA Errors**
   - Verify CUDA installation
   - Check GPU compatibility
   - Fall back to CPU mode

4. **Streamlit Issues**
   - Clear browser cache
   - Restart the application
   - Check port availability

### Performance Issues

- **Slow Processing**: Enable GPU acceleration
- **High Memory Usage**: Use smaller models
- **Poor Quality**: Adjust parameters or try different models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run code formatting
black .
flake8 .
mypy .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Streamlit](https://streamlit.io/) for the web framework
- [PyTorch](https://pytorch.org/) for the deep learning framework
- The open-source community for the pre-trained models

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## Future Enhancements

- [ ] Support for more languages
- [ ] Custom model training
- [ ] API endpoint for external access
- [ ] Advanced analytics dashboard
- [ ] Integration with cloud services
- [ ] Mobile app development


# Text-Summarization-Model
