# House Price Prediction System

A modular machine learning pipeline for predicting French real estate prices with 82% accuracy. This system helps investors identify good, fair, or bad investment opportunities by comparing listing prices against predicted fair market values.

## 🎯 Project Overview

This project uses historical French property transaction data to train a Random Forest regression model that predicts house prices based on location, size, rooms, and other features. The system features a production-ready ML pipeline with:

- **Modular Architecture**: Separate components for data ingestion, validation, and transformation
- **Data Validation**: Schema validation and drift detection
- **Feature Engineering**: 19 engineered features including target encoding and interaction terms
- **MongoDB Integration**: Scalable data storage and retrieval
- **Production Pipeline**: Automated end-to-end ML pipeline

The model can be used to:
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
│   ├── components/            # Pipeline components
│   │   ├── data_ingestion.py      # MongoDB data ingestion
│   │   ├── data_validation.py      # Data validation & drift detection
│   │   └── data_transformation.py  # Feature engineering & preprocessing
│   ├── entity/                # Data classes for configs & artifacts
│   │   ├── config_entity.py       # Configuration entities
│   │   └── artifact_entity.py      # Pipeline artifacts
│   ├── constants/             # Configuration constants
│   │   └── training_pipeline/      # Pipeline constants
│   ├── data_schema/           # Data validation schemas
│   │   └── schema.yaml            # Data schema definition
│   ├── exception/             # Custom exceptions
│   │   └── exception.py
│   ├── logging/               # Logging configuration
│   │   └── logger.py
│   └── utils/                 # Utility functions
│       └── main_utils/
│           └── utils.py          # YAML, numpy, pickle utilities
├── notebooks/                 # Jupyter notebooks
│   └── test.ipynb            # Model training and evaluation
├── house_data/                # Raw data storage
│   └── filtered_data.csv      # Main dataset (650K+ records)
├── Artifact/                  # Pipeline artifacts (auto-generated)
├── final_model/               # Trained models & preprocessors
├── logs/                      # Application logs
├── main.py                    # Pipeline testing script
├── push_data.py               # MongoDB data upload script
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── Dockerfile                 # Docker configuration
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🚀 Features

### Pipeline Components

**1. Data Ingestion**
- Fetch data from MongoDB
- Export to feature store
- Train/test split (80/20)
- Batch processing for large datasets

**2. Data Validation**
- Schema validation (column count, data types)
- Numerical column validation
- Data drift detection using Kolmogorov-Smirnov test
- Drift report generation (YAML)

**3. Data Transformation**
- Data cleaning (price formatting, outlier removal)
- Feature engineering (19 engineered features):
  - Location-based features (commune/department average prices)
  - Property characteristics (size, rooms, land ratio)
  - Interaction features (size × property type)
  - Non-linear features (squared transformations)
  - Temporal features (month, season)
- Target encoding (train-based, no data leakage)
- Scaling and imputation (StandardScaler + SimpleImputer)
- Save transformed data as numpy arrays

### Data Processing
- **Data Cleaning**: Handle missing values, remove outliers, clean price formats
- **Feature Engineering**: 19 engineered features including target encoding
- **Target Encoding**: Commune and department average prices (calculated on train set)
- **Data Validation**: Schema validation and drift detection
- **Preprocessing**: Scaling and imputation pipeline

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

### Setup MongoDB Connection

1. Create a `.env` file in the project root:
```bash
MONGO_DB_URL=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?appName=<appname>
```

2. Upload data to MongoDB:
```bash
python push_data.py
```

This will upload `filtered_data.csv` to MongoDB in batches.

### Running the Pipeline

Test the complete pipeline:

```bash
python main.py
```

This will run:
1. **Data Ingestion**: Fetch data from MongoDB, split train/test
2. **Data Validation**: Validate schema, check drift
3. **Data Transformation**: Clean, feature engineer, scale data

The pipeline creates artifacts in the `Artifact/` directory with timestamps.

### Training the Model

Run the training notebook:

```bash
jupyter notebook notebooks/test.ipynb
```

The notebook includes:
1. Data loading and cleaning
2. Feature engineering (19 features)
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

### Training Features (19 total)
1. **Surface reelle bati** - Built surface area
2. **Nombre pieces principales** - Number of rooms
3. **Surface terrain** - Land surface area
4. **land_ratio** - Land to built surface ratio
5. **total_surface** - Total property surface
6. **room_density** - Rooms per square meter
7. **is_house** - Binary: 1 for Maison, 0 for Appartement
8. **is_large** - Binary: 1 if >150m²
9. **is_small** - Binary: 1 if <60m²
10. **month** - Month of sale
11. **season** - Season of sale (Winter/Spring/Summer/Fall)
12. **Code departement** - Department code
13. **commune_avg_price** - Average price in commune (target encoded)
14. **dept_avg_price** - Average price in department (target encoded)
15. **commune_price_rank** - Relative price position
16. **size_x_house** - Interaction: size × property type
17. **rooms_x_house** - Interaction: rooms × property type
18. **dept_squared** - Department code squared (non-linear)
19. **surface_squared** - Built surface squared (non-linear)

### Engineered Features
- **Location-based**: Commune/department average prices (target encoding), price ranks
- **Property characteristics**: Land ratio, room density, size categories
- **Interaction features**: Size × property type, rooms × property type
- **Non-linear features**: Squared transformations for complex relationships
- **Temporal features**: Month, season of sale
- **Target Encoding**: Calculated on train set to prevent data leakage

## 🔮 Future Enhancements

### Completed Features ✅
- [x] **Modular Pipeline Architecture**: Data ingestion, validation, transformation components
- [x] **MongoDB Integration**: Scalable data storage and retrieval
- [x] **Data Validation**: Schema validation and drift detection
- [x] **Feature Engineering**: 19 engineered features with target encoding
- [x] **Data Preprocessing**: Scaling and imputation pipeline
- [x] **Custom Exception Handling**: Structured error handling with logging
- [x] **Artifact Management**: Timestamped pipeline artifacts

### Planned Features
- [ ] **Model Training Component**: Automated model training pipeline
- [ ] **Model Evaluation Component**: Comprehensive model evaluation metrics
- [ ] **Model Trainer**: Train and save Random Forest model
- [ ] **Property Condition Data**: Age, renovation status, condition scores
- [ ] **Neighborhood Quality**: School ratings, crime rates, amenities
- [ ] **Rental Yield**: Calculate potential rental income and ROI
- [ ] **Market Timing**: Interest rates, price trends, economic indicators
- [ ] **Web Interface**: FastAPI-based prediction API
- [ ] **Real-time Updates**: Automated model retraining based on drift
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
