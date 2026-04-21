# House Price Prediction System

A machine learning project for predicting French real estate prices with 82% accuracy. This system helps investors identify good, fair, or bad investment opportunities by comparing listing prices against predicted fair market values.

## 🎯 Project Overview

This project uses historical French property transaction data to train a Random Forest regression model that predicts house prices based on location, size, rooms, and other features. The model can be used to:

- **Predict fair market value** of properties
- **Identify undervalued properties** (good investment opportunities)
- **Detect overpriced listings** (avoid bad investments)
- **Provide data-driven insights** for real estate decisions

## 📊 Model Performance

- **R² Score**: 82% (Excellent for real estate prediction)
- **MAE**: Mean Absolute Error on test set
- **RMSE**: Root Mean Squared Error on test set
- **Model**: Random Forest Regressor (100 estimators, max depth 15)

## 🏗️ Project Structure

```
house-prediction/
├── house_prediction/          # Main package
│   ├── __init__.py
│   ├── pipeline/              # ML pipeline components
│   ├── components/            # Data processing components
│   ├── cloud/                 # Cloud integration (MLflow, DagsHub)
│   ├── constants/             # Configuration constants
│   ├── data_schema/           # Data validation schemas
│   ├── exception/             # Custom exceptions
│   ├── logging/               # Logging configuration
│   └── utils/                 # Utility functions
├── notebooks/                 # Jupyter notebooks
│   └── test.ipynb            # Model training and evaluation
├── house_data/                # Raw data storage
├── valid_data/                # Validation data
├── prediction_output/         # Model predictions
├── templates/                 # HTML templates (if web app)
├── filtered_data.csv          # Main dataset (650K+ records)
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── Dockerfile                 # Docker configuration
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🚀 Features

### Data Processing
- **Data Cleaning**: Handle missing values, remove outliers, clean price formats
- **Feature Engineering**: 18+ engineered features including:
  - Location-based features (commune/department average prices)
  - Property characteristics (size, rooms, land ratio)
  - Interaction features (size × property type)
  - Non-linear features (squared transformations)
  - Temporal features (month, season)

### Machine Learning
- **Model**: Random Forest Regressor
- **Training**: 80/20 train-test split
- **Cross-validation**: 5-fold CV for robustness
- **Feature Importance**: Identifies key price drivers

### Prediction Capabilities
- **Price Prediction**: Estimate fair market value
- **Investment Analysis**: Classify as good/fair/bad deal
- **Feature Analysis**: Understand what drives prices

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd house-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package**
```bash
pip install -e .
```

## 📊 Dataset

The project uses French real estate transaction data (`filtered_data.csv`) with:

- **650,000+ records** of property transactions
- **10 original features**:
  - Date mutation (transaction date)
  - Nature mutation (transaction type)
  - Valeur fonciere (property price)
  - Code postal (postal code)
  - Commune (city/town)
  - Code departement (department code)
  - Type local (property type: Maison/Appartement)
  - Surface reelle bati (built surface area)
  - Nombre pieces principales (number of rooms)
  - Surface terrain (land surface area)

## 🔧 Usage

### Training the Model

Run the training notebook:

```bash
jupyter notebook notebooks/test.ipynb
```

The notebook includes:
1. Data loading and cleaning
2. Feature engineering (18+ features)
3. Train/test split
4. Model training (Random Forest)
5. Performance evaluation
6. Feature importance analysis
7. Visualization plots

### Making Predictions

After training, use the pipeline to predict prices:

```python
from sklearn.pipeline import Pipeline
import pandas as pd

# Load your trained pipeline
pipeline = joblib.load('model.pkl')

# Prepare input data
input_data = {
    'Surface reelle bati': 120,
    'Nombre pieces principales': 5,
    'Surface terrain': 500,
    # ... other features
}

# Make prediction
predicted_price = pipeline.predict(input_data)
print(f"Predicted price: €{predicted_price[0]:,.2f}")
```

### Investment Recommendation

Compare listing price vs predicted price:

```python
listing_price = 320000
predicted_price = 350000

price_diff_pct = ((listing_price - predicted_price) / predicted_price) * 100

if price_diff_pct < -10:
    recommendation = "GOOD DEAL - Undervalued"
elif price_diff_pct <= 10:
    recommendation = "FAIR PRICE - Reasonably priced"
else:
    recommendation = "OVERPRICED - Consider negotiating"
```

## 🛠️ Tech Stack

- **Machine Learning**: scikit-learn, pandas, numpy
- **Data Visualization**: matplotlib, seaborn
- **MLOps**: MLflow, DagsHub
- **API**: FastAPI, uvicorn
- **Database**: MongoDB (pymongo)
- **Containerization**: Docker
- **Version Control**: Git

## 📈 Model Features

### Top Predictive Features
1. **commune_avg_price** - Average price in the commune
2. **Surface reelle bati** - Built surface area
3. **dept_avg_price** - Average price in the department
4. **commune_price_rank** - Relative price position
5. **total_surface** - Total property surface

### Engineered Features
- **Location-based**: Commune/department average prices, price ranks
- **Property characteristics**: Land ratio, room density, size categories
- **Interaction features**: Size × property type, rooms × property type
- **Non-linear features**: Squared transformations for complex relationships
- **Temporal features**: Month, season of sale

## 🔮 Future Enhancements

### Planned Features
- [ ] **Property Condition Data**: Age, renovation status, condition scores
- [ ] **Neighborhood Quality**: School ratings, crime rates, amenities
- [ ] **Rental Yield**: Calculate potential rental income and ROI
- [ ] **Market Timing**: Interest rates, price trends, economic indicators
- [ ] **Web Interface**: FastAPI-based prediction API
- [ ] **Real-time Updates**: Automated model retraining
- [ ] **Geospatial Analysis**: Map-based visualizations

### Data Sources to Integrate
- Government open data (schools, crime, demographics)
- Real estate APIs for market trends
- Satellite imagery for property condition
- Economic indicators (interest rates, inflation)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

- **Your Name** - Initial work

## 🙏 Acknowledgments

- French government for open real estate data
- scikit-learn community for excellent ML tools
- MLflow and DagsHub for MLOps tools

## 📞 Contact

For questions or suggestions, please open an issue or contact [diallisouleyman78@gmail.com]

---

**Note**: This model provides price estimates based on historical data. Always conduct thorough due diligence including property inspections, legal checks, and market analysis before making investment decisions.
