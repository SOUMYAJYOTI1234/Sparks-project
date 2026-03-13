# 🎓 Student Score Prediction — End-to-End ML Project

An end-to-end machine learning project that predicts student exam scores based on daily study hours. Built with a modular production architecture, Flask web interface, and Docker support.

## 📸 Project Overview

| Feature | Description |
|---------|-------------|
| **Algorithm** | Compares 8 regression models, selects the best |
| **Best Model** | Auto-selected based on R² score |
| **Web App** | Flask-based prediction interface |
| **Pipeline** | Modular training & prediction pipelines |
| **Logging** | Timestamped log files with detailed tracing |
| **Docker** | Containerized for easy deployment |

## 🏗️ Project Structure

```
Student_price_prediction/
├── src/
│   └── student_score_prediction/
│       ├── __init__.py
│       ├── logger.py              # Custom logging
│       ├── exception.py           # Custom exception handling
│       ├── utils.py               # Utility functions
│       ├── components/
│       │   ├── data_ingestion.py      # Data download & split
│       │   ├── data_transformation.py # Feature preprocessing
│       │   └── model_trainer.py       # Model comparison & training
│       └── pipeline/
│           ├── training_pipeline.py   # Orchestrates training
│           └── prediction_pipeline.py # Serves predictions
├── templates/
│   └── index.html                 # Web UI
├── notebook/
│   └── Sparks.ipynb               # Original EDA notebook
├── artifacts/                     # Generated models & data (gitignored)
├── logs/                          # Log files (gitignored)
├── app.py                         # Flask web application
├── setup.py                       # Package configuration
├── requirements.txt               # Dependencies
├── Dockerfile                     # Docker configuration
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/SOUMYAJYOTI1234/Sparks-project.git
cd Sparks-project

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Run the Training Pipeline

```bash
python -m src.student_score_prediction.pipeline.training_pipeline
```

This will:
1. Download the dataset from the source
2. Split into train/test sets (80/20)
3. Apply StandardScaler preprocessing
4. Compare 8 regression models
5. Select and save the best model

### Run the Web Application

```bash
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

### Docker

```bash
# Build the image
docker build -t student-score-predictor .

# Run the container
docker run -p 5000:5000 student-score-predictor
```

## 🤖 Models Compared

| Model | Description |
|-------|-------------|
| Linear Regression | Standard OLS regression |
| Ridge | L2 regularized regression |
| Lasso | L1 regularized regression |
| ElasticNet | L1 + L2 regularization |
| Decision Tree | Non-linear tree-based model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting ensemble |
| SVR | Support vector regression |

## 📊 Dataset

- **Source**: [Student Scores Dataset](https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv)
- **Features**: `Hours` (study hours per day)
- **Target**: `Scores` (exam score 0-100)
- **Samples**: 25 records

## 🙏 Acknowledgments

This project is completed as part of the **Data Science & Business Analytics Internship** by [The Sparks Foundation](https://www.thesparksfoundationsingapore.org/).

---
*Built by [Soumyajyoti Chatterjee](https://github.com/SOUMYAJYOTI1234)*
