# 📰 News Category Classification

A machine learning project that automatically classifies news articles into categories (World, Sports, Business, Sci/Tech) using transformer models and provides an interactive web interface.

## 🚀 Features

- **Real-time Classification**: Classify news articles into 4 categories instantly
- **Interactive Web App**: User-friendly Streamlit interface with examples
- **Transformer Models**: Uses DistilBERT for accurate text classification
- **Training Pipeline**: Complete training script for custom model fine-tuning
- **Fast Performance**: Optimized for quick predictions and model loading

## 📋 Categories

- 🌍 **World**: International news, politics, global events
- ⚽ **Sports**: Football, basketball, championships, athlete news
- 💼 **Business**: Finance, markets, corporate news, economy
- 🔬 **Sci/Tech**: Technology, science discoveries, innovations

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/news-classification.git
   cd news-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```bash
   streamlit run src/app.py
   ```

## 📦 Dependencies

Create a `requirements.txt` file with:

```
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
datasets>=2.14.0
accelerate>=0.26.0
numpy>=1.23.5,<2.3.0
pandas>=1.5.0
```

## 🎯 Usage

### Web Application

1. **Start the app**:
   ```bash
   streamlit run src/app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Enter news content**:
   - Add a news title
   - Add a description
   - Click "🔍 Predict Category"

4. **View results**:
   - See the predicted category with confidence score
   - Expand "📊 See all category probabilities" for detailed breakdown

### Example Usage

**Input:**
- **Title**: "PSG sinks Barcelona with late winner in Champions League"
- **Description**: "PSG's trip to Barcelona was always seen as the standout clash of matchday two..."

**Output:**
- **Predicted Category**: ⚽ Sports
- **Confidence**: 95.2% (High)

## 🏗️ Project Structure

```
ML Project/
├── src/
│   ├── app.py                 # 🌐 Streamlit web application
│   ├── transformer_train.py   # 🤖 Transformer model training
│   ├── baseline_model.py      # 📊 Baseline models (LogReg, NB)
│   ├── inference.py           # 🔍 Simple inference script
│   ├── preprocess.py          # 🧹 Data preprocessing
│   ├── eda.py                # 📈 Exploratory data analysis
│   └── evaluate_model.py     # 📋 Model evaluation
├── data/
│   ├── train.csv             # 📚 Training dataset (AG News)
│   ├── test.csv              # 🧪 Test dataset
│   └── processed/            # 🔄 Processed data files
│       ├── train_final.csv
│       ├── val_final.csv
│       └── test_processed.csv
├── models/
│   └── distilbert_news/      # 🏠 Model storage
│       ├── final/            # Pre-trained model
│       ├── final_trained/    # Custom trained model
│       └── trained_model/    # Training checkpoints
├── logs/                     # 📝 Training logs
├── .cache/                   # 💾 Model cache
├── requirements.txt          # 📦 Python dependencies
└── README.md                # 📖 Project documentation
```

## 🤖 Model Details

### Current Model
- **Base Model**: `textattack/distilbert-base-uncased-ag-news`
- **Architecture**: DistilBERT (Distilled BERT)
- **Training Data**: AG News Dataset
- **Categories**: 4 classes (World, Sports, Business, Sci/Tech)
- **Performance**: ~94% accuracy on AG News test set

### Available Models

#### 1. 🤖 Transformer Model (Primary)
- **Script**: `src/transformer_train.py`
- **Model**: DistilBERT fine-tuned on AG News
- **Accuracy**: ~94%
- **Features**: Multi-process tokenization, early stopping, GPU support

#### 2. 📊 Baseline Models
- **Script**: `src/baseline_model.py`
- **Models**: Logistic Regression, Multinomial Naive Bayes
- **Features**: TF-IDF vectorization, fast training
- **Purpose**: Comparison baseline for transformer models

#### 3. 🔍 Simple Inference
- **Script**: `src/inference.py`
- **Purpose**: Quick predictions without web interface
- **Usage**:
  ```python
  from src.inference import predict
  category, confidence = predict("Your news text here")
  print(f"Category: {category}, Confidence: {confidence:.2f}")
  ```

### Training Your Own Model

#### Transformer Training
```bash
python src/transformer_train.py
```

#### Baseline Model Training
```bash
python src/baseline_model.py
```

#### Data Processing Pipeline
```bash
python src/preprocess.py    # Clean and prepare data
python src/eda.py          # Exploratory data analysis
python src/evaluate_model.py # Model evaluation
```

## 🎨 Web Interface Features

- **🔍 Smart Prediction**: Real-time classification with confidence scores
- **📊 Detailed Analytics**: View probabilities for all categories
- **💡 Quick Examples**: Pre-filled examples for testing
- **🎯 Color-coded Results**: Visual confidence indicators
- **⚡ Fast Loading**: Cached model loading for quick startup

## 📊 Performance

- **Model Loading**: ~10-30 seconds (first time), ~2-5 seconds (cached)
- **Prediction Speed**: <1 second per article
- **Accuracy**: ~94% on AG News dataset
- **Memory Usage**: ~500MB RAM

## 🔧 Configuration

### Model Settings
- **Max Token Length**: 128 tokens
- **Batch Size**: Optimized for single predictions
- **Device**: Auto-detects CUDA/CPU

### Customization
- Modify `src/app.py` to change UI elements
- Update model in `load_model()` function for different models
- Adjust confidence thresholds in prediction logic

## 🚀 Deployment

### Local Development
```bash
streamlit run src/app.py
```

### Production Deployment
- **Streamlit Cloud**: Connect your GitHub repo
