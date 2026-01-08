# Gastronomie

Streamlit application for analyzing food trends from recipe and user interaction data.

## 🚀 Getting Started

### 1. Download the data

Download the dataset and extract `RAW_interactions.csv` and `RAW_recipes.csv` into the `data/` folder:

📥 [Download data.zip](https://cdn.discordapp.com/attachments/950155154659889152/1458637326736425204/data.zip?ex=69605d5e&is=695f0bde&hm=f718b392982fcd0c72b4a3c8794aec64418482fd5da4df61c201783ce7cf9256&)

### 2. Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate environment

# Windows (Git Bash)
source .venv/Scripts/activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

## 📊 Menus

| Menu | Description |
|------|-------------|
| **Overview** | Global stats (interactions, recipes, users) + monthly trends |
| **Recipes trends** | Ranking of most popular recipes with filters |
| **Ingredient trends** | Ingredient popularity evolution over time |
| **Predictions** | Future trend predictions (linear regression) |
| **Consumer expectations (topics)** | Topic analysis (LDA) to identify themes |
