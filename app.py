"""
Streamlit Web UI for Text Summarization Model
=============================================

A modern, interactive web interface for the text summarization system.
"""

import streamlit as st
import json
import time
from pathlib import Path
import sys

# Add the current directory to Python path to import our modules
sys.path.append(str(Path(__file__).parent))

from transformers import pipeline
import torch

# Page configuration
st.set_page_config(
    page_title="Text Summarization Model",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .summary-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_summarization_model(model_name="facebook/bart-large-cnn"):
    """Load the summarization model with caching."""
    try:
        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        return summarizer, None
    except Exception as e:
        return None, str(e)

def load_sample_texts():
    """Load sample texts from the mock database."""
    try:
        db_path = Path("sample_texts.json")
        if db_path.exists():
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("articles", [])
        return []
    except Exception as e:
        st.error(f"Error loading sample texts: {e}")
        return []

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📝 Text Summarization Model</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_options = {
        "BART Large CNN": "facebook/bart-large-cnn",
        "BART Base CNN": "facebook/bart-base-cnn",
        "T5 Small": "t5-small",
        "T5 Base": "t5-base",
        "Pegasus CNN": "google/pegasus-cnn_dailymail"
    }
    
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        options=list(model_options.keys()),
        index=0
    )
    
    model_name = model_options[selected_model]
    
    # Summarization parameters
    st.sidebar.subheader("📊 Summarization Parameters")
    
    max_length = st.sidebar.slider(
        "Maximum Summary Length:",
        min_value=20,
        max_value=200,
        value=100,
        help="Maximum number of tokens in the summary"
    )
    
    min_length = st.sidebar.slider(
        "Minimum Summary Length:",
        min_value=10,
        max_value=100,
        value=30,
        help="Minimum number of tokens in the summary"
    )
    
    do_sample = st.sidebar.checkbox(
        "Enable Sampling:",
        value=False,
        help="Enable sampling for more diverse summaries"
    )
    
    temperature = st.sidebar.slider(
        "Temperature:",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Controls randomness in generation"
    ) if do_sample else 1.0
    
    # Load model
    with st.spinner(f"Loading {selected_model}..."):
        summarizer, error = load_summarization_model(model_name)
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.stop()
    
    st.success(f"✅ {selected_model} loaded successfully!")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📝 Summarize Text", "📚 Sample Articles", "ℹ️ About"])
    
    with tab1:
        st.header("Text Summarization")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Upload File"],
            horizontal=True
        )
        
        text_input = ""
        
        if input_method == "Type/Paste Text":
            text_input = st.text_area(
                "Enter text to summarize:",
                height=200,
                placeholder="Paste your text here...",
                help="Text should be at least 50 characters long for best results"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt', 'md'],
                help="Upload a .txt or .md file"
            )
            
            if uploaded_file:
                try:
                    text_input = str(uploaded_file.read(), "utf-8")
                    st.success("File uploaded successfully!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Summarize button
        if st.button("🚀 Generate Summary", type="primary"):
            if not text_input or len(text_input.strip()) < 50:
                st.error("Please enter text that is at least 50 characters long.")
            else:
                with st.spinner("Generating summary..."):
                    try:
                        start_time = time.time()
                        
                        # Generate summary
                        summary_result = summarizer(
                            text_input,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=do_sample,
                            temperature=temperature
                        )
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        summary_text = summary_result[0]['summary_text']
                        
                        # Display results
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Original Length", f"{len(text_input):,} characters")
                            st.metric("Summary Length", f"{len(summary_text):,} characters")
                            st.metric("Compression Ratio", f"{len(summary_text)/len(text_input)*100:.1f}%")
                            st.metric("Processing Time", f"{processing_time:.2f} seconds")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                            st.subheader("📝 Generated Summary")
                            st.write(summary_text)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download option
                        st.download_button(
                            label="📥 Download Summary",
                            data=summary_text,
                            file_name="summary.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error(f"❌ Error generating summary: {e}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Sample Articles")
        
        # Load sample articles
        articles = load_sample_texts()
        
        if not articles:
            st.warning("No sample articles found. Please run the main script first to create sample data.")
        else:
            # Article selection
            article_titles = [f"{article['title']} ({article['category']})" for article in articles]
            selected_article_idx = st.selectbox(
                "Select an article:",
                range(len(articles)),
                format_func=lambda x: article_titles[x]
            )
            
            selected_article = articles[selected_article_idx]
            
            # Display article
            st.subheader(f"📖 {selected_article['title']}")
            st.caption(f"Category: {selected_article['category']} | ID: {selected_article['id']}")
            
            st.text_area(
                "Article Content:",
                value=selected_article['content'],
                height=200,
                disabled=True
            )
            
            # Quick summarize button
            if st.button("🚀 Summarize This Article", type="primary"):
                with st.spinner("Generating summary..."):
                    try:
                        start_time = time.time()
                        
                        summary_result = summarizer(
                            selected_article['content'],
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=do_sample,
                            temperature=temperature
                        )
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        summary_text = summary_result[0]['summary_text']
                        
                        # Display summary
                        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                        st.subheader("📝 Summary")
                        st.write(summary_text)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Length", f"{len(selected_article['content']):,} chars")
                        with col2:
                            st.metric("Summary Length", f"{len(summary_text):,} chars")
                        with col3:
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {e}")
    
    with tab3:
        st.header("About This Application")
        
        st.markdown("""
        ### 🚀 Advanced Text Summarization System
        
        This application demonstrates a modern text summarization system using state-of-the-art transformer models.
        
        #### ✨ Features:
        - **Multiple Models**: Support for BART, T5, and Pegasus models
        - **Configurable Parameters**: Adjustable summary length, temperature, and sampling
        - **Interactive UI**: User-friendly interface with real-time feedback
        - **Sample Data**: Built-in sample articles for testing
        - **Performance Metrics**: Processing time and compression ratio tracking
        
        #### 🔧 Technical Details:
        - Built with **Streamlit** for the web interface
        - Powered by **Hugging Face Transformers**
        - Supports **GPU acceleration** when available
        - **Caching** for improved performance
        
        #### 📊 Supported Models:
        - **BART Large CNN**: Best for news articles and general text
        - **BART Base CNN**: Faster alternative with good quality
        - **T5 Small/Base**: Google's Text-to-Text Transfer Transformer
        - **Pegasus CNN**: Specialized for news summarization
        
        #### 🎯 Use Cases:
        - News article summarization
        - Academic paper abstracts
        - Meeting notes compression
        - Content curation
        - Research assistance
        
        #### 💡 Tips for Best Results:
        - Use text that is at least 50 characters long
        - Adjust max/min length based on your needs
        - Enable sampling for more diverse summaries
        - Try different models for different content types
        """)
        
        # System information
        st.subheader("🖥️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("PyTorch Version", torch.__version__)
            st.metric("CUDA Available", "Yes" if torch.cuda.is_available() else "No")
        
        with col2:
            if torch.cuda.is_available():
                st.metric("GPU Device", torch.cuda.get_device_name(0))
                st.metric("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

if __name__ == "__main__":
    main()
