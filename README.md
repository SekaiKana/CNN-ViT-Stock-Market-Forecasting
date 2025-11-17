# Stock Movement Prediction with CNN-ViT on Candlestick Charts

This project explores the use of a hybrid Convolutional Neural Network (CNN) and Vision Transformer (ViT) model to analyze and predict stock price movements. The core methodology treats financial prediction as a computer vision problem by training the model on images of candlestick charts rather than raw time-series data.

## 📋 Overview

The repository contains two primary experiments:

1. **Regression Model**: An initial attempt to predict the precise 5-day future log-return of a stock
2. **Classification Model**: A pivot to a binary classification task to predict whether a stock's price will move up or down over the 5-day forecast horizon

## 📊 Methodology

The entire workflow is built on the idea of converting 1-dimensional time-series data into 2-dimensional image data to leverage powerful computer vision architectures.

### Data Collection

- OHLCV (Open, High, Low, Close, Volume) data for four major tech stocks (META, AAPL, MSFT, GOOGL) was downloaded using `yfinance`
- Data spans from January 1, 2015, to October 1, 2025
- Saved in `market_data.csv`

### Image Generation

- A sliding window approach creates samples, each with a 120-day lookback window
- Using `mplfinance`, each 120-day window is rendered as a 400x600px candlestick chart image (including volume bars)
- Images are cached as `.npz` files to `chart_cache/` (regression) and `chart_cache_class/` (classification) directories to speed up training

### Target Generation

- **For Regression**: Target is the 5-day future log-return, calculated as $\log(\frac{\text{Price}_{t+5}}{\text{Price}_t})$
- **For Classification**: Target is a binary class (0 or 1), representing whether the 5-day future log-return was negative or positive

## 🧠 Model Architecture: CNN-ViT Hybrid

Both experiments use the same hybrid model architecture, combining the strengths of CNNs and Transformers:

### CNN Backbone (ConvNet)
A custom CNN acts as a powerful feature extractor, processing the raw candlestick chart image through a series of Convolution, BatchNorm, and MaxPool layers to generate high-level feature maps.

### Patch Embedding (PatchEmbed)
A 1x1 Convolution layer flattens the CNN's feature maps into a 1D sequence of patches (tokens), making it suitable for a Transformer.

### Vision Transformer (ViT)
- A special `[CLS]` token is prepended to the sequence of patches
- Positional embeddings are added to retain spatial information
- The sequence is processed by a stack of `TransformerBlock` modules, which use Multi-Head Self-Attention (MHSA) to learn relationships between different parts of the chart
- The final output state of the `[CLS]` token is passed to a linear head for the final prediction

## 📈 Experiments & Results

### Experiment 1: Regression (Price Prediction)

**File**: `data.ipynb`

**Goal**: Predict the exact 5-day future log-return

**Loss Function**: Trained with both MSE and Huber Loss

**Results**: This task proved to be extremely difficult. The best model (using Huber Loss) achieved a test Root Mean Square Error (RMSE) of ~0.0331, or a 3.3% log-return error. The R² score was negative, indicating the model performed worse than a simple baseline (e.g., predicting the mean return). This highlights the inefficiency of trying to predict exact future prices from charts alone.

### Experiment 2: Classification (Directional Prediction)

**File**: `classification_model.ipynb`

**Goal**: Pivot to a simpler, more realistic binary classification task (will the price be higher or lower in 5 days)

**Data Handling**:
- Per-ticker, sequential train/validation split (80/20) used to prevent data leakage
- Undersampling applied to both training and validation sets to create a balanced 50/50 class distribution

**Loss Function**: FocalLoss used to help the model focus on harder-to-classify examples

**Results**: This approach yielded more promising results. The model achieved a peak validation accuracy of ~54.1%. While this is a small edge, a consistent accuracy above 50% is significant in financial markets. The final confusion matrix confirms that the model performs slightly better than a random 50/50 guess on both "up" and "down" classes.

## 🚀 Getting Started

### Prerequisites

```bash
pip install yfinance mplfinance torch torchvision numpy pandas
```

### Running the Experiments

1. **Regression Model**: Open and run `data.ipynb`
2. **Classification Model**: Open and run `classification_model.ipynb`

## 📁 Project Structure

```
.
├── data.ipynb                    # Regression experiment
├── classification_model.ipynb    # Classification experiment
├── market_data.csv              # OHLCV data for stocks
├── chart_cache/                 # Cached images for regression
└── chart_cache_class/           # Cached images for classification
```

## 🔍 Key Findings

- **Regression**: Predicting exact future prices from candlestick charts alone is extremely challenging and yielded poor results
- **Classification**: Binary directional prediction shows more promise, achieving ~54% accuracy - a small but potentially significant edge in financial markets
- **Approach**: Treating stock prediction as a computer vision problem is viable, particularly for directional forecasting rather than precise price prediction

