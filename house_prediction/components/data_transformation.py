import sys, os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from house_prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from house_prediction.entity.config_entity import DataTransformationConfig
from house_prediction.utils.main_utils.utils import save_numpy_array_data, save_object
from house_prediction.exception.exception import HousePredictionException
from house_prediction.logging.logger import logging
from house_prediction.constants.training_pipeline import TARGET_COLUMN


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise HousePredictionException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise HousePredictionException(e, sys)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        try:
            # Clean price column (remove commas and convert to float)
            df['Valeur fonciere'] = df['Valeur fonciere'].astype(str).str.replace(',', '').str.replace('"', '').astype(float)

            # Convert date to datetime
            df['Date mutation'] = pd.to_datetime(df['Date mutation'], dayfirst=True)

            # Handle missing values
            df = df.dropna()

            # Convert department code to numeric
            df['Code departement'] = pd.to_numeric(df['Code departement'], errors='coerce')
            df = df.dropna(subset=['Code departement'])
            df['Code departement'] = df['Code departement'].astype(int)

            # Remove outliers (prices beyond 3 standard deviations)
            mean_price = df['Valeur fonciere'].mean()
            std_price = df['Valeur fonciere'].std()
            df = df[(df['Valeur fonciere'] >= mean_price - 3*std_price) &
                    (df['Valeur fonciere'] <= mean_price + 3*std_price)]

            return df
        except Exception as e:
            raise HousePredictionException(e, sys)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering from test.ipynb"""
        try:
            # Land ratio
            df['land_ratio'] = df['Surface terrain'] / df['Surface reelle bati']

            # Total surface
            df['total_surface'] = df['Surface reelle bati'] + df['Surface terrain']

            # Room density
            df['room_density'] = df['Nombre pieces principales'] / df['Surface reelle bati']

            # Property type binary (1 for Maison, 0 for Appartement)
            df['is_house'] = (df['Type local'] == 'Maison').astype(int)

            # Large property flag (>150m²)
            df['is_large'] = (df['Surface reelle bati'] > 150).astype(int)

            # Small property flag (<60m²)
            df['is_small'] = (df['Surface reelle bati'] < 60).astype(int)

            # Month of sale
            df['month'] = df['Date mutation'].dt.month

            # Season (Winter:12-2, Spring:3-5, Summer:6-8, Fall:9-11)
            def get_season(month):
                if month in [12, 1, 2]:
                    return 0  # Winter
                elif month in [3, 4, 5]:
                    return 1  # Spring
                elif month in [6, 7, 8]:
                    return 2  # Summer
                else:
                    return 3  # Fall

            df['season'] = df['month'].apply(get_season)

            # Price per m²
            df['price_per_m2'] = df['Valeur fonciere'] / df['Surface reelle bati']

            # Price per room
            df['price_per_room'] = df['Valeur fonciere'] / df['Nombre pieces principales']

            # Terrain to bati ratio
            df['terrain_bati_ratio'] = df['Surface terrain'] / df['Surface reelle bati']

            # Interaction: size x house type
            df['size_x_house'] = df['Surface reelle bati'] * df['is_house']

            # Interaction: rooms x house type
            df['rooms_x_house'] = df['Nombre pieces principales'] * df['is_house']

            # Department squared (non-linear effect)
            df['dept_squared'] = df['Code departement'] ** 2

            # Surface bati squared (non-linear effect)
            df['surface_squared'] = df['Surface reelle bati'] ** 2

            return df
        except Exception as e:
            raise HousePredictionException(e, sys)

    def target_encoding(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """Apply target encoding calculated on train set"""
        try:
            # Calculate target encoding on train set
            commune_avg_price = train_df.groupby('Commune')['Valeur fonciere'].mean()
            dept_avg_price = train_df.groupby('Code departement')['Valeur fonciere'].mean()

            # Apply to train set
            train_df['commune_avg_price'] = train_df['Commune'].map(commune_avg_price)
            train_df['dept_avg_price'] = train_df['Code departement'].map(dept_avg_price)
            train_df['commune_price_rank'] = train_df['commune_avg_price'].rank(pct=True)

            # Apply to test set
            test_df['commune_avg_price'] = test_df['Commune'].map(commune_avg_price)
            test_df['dept_avg_price'] = test_df['Code departement'].map(dept_avg_price)
            test_df['commune_price_rank'] = test_df['commune_avg_price'].rank(pct=True)

            # Fill missing values
            mean_price = train_df['Valeur fonciere'].mean()
            train_df['commune_avg_price'] = train_df['commune_avg_price'].fillna(mean_price)
            train_df['dept_avg_price'] = train_df['dept_avg_price'].fillna(mean_price)
            train_df['commune_price_rank'] = train_df['commune_price_rank'].fillna(0.5)

            test_df['commune_avg_price'] = test_df['commune_avg_price'].fillna(mean_price)
            test_df['dept_avg_price'] = test_df['dept_avg_price'].fillna(mean_price)
            test_df['commune_price_rank'] = test_df['commune_price_rank'].fillna(0.5)

            return train_df, test_df
        except Exception as e:
            raise HousePredictionException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """Create a pipeline with scaler and imputer"""
        try:
            logging.info("Creating data transformer pipeline")
            processor = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            logging.info("Data transformer pipeline created")
            return processor
        except Exception as e:
            raise HousePredictionException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Initiating data transformation")
        try:
            logging.info("Reading validated train and test data")
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info("Cleaning data")
            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)

            logging.info("Applying feature engineering")
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            logging.info("Applying target encoding")
            train_df, test_df = self.target_encoding(train_df, test_df)

            # Select features for modeling
            training_features = [
                'Surface reelle bati', 'Nombre pieces principales', 'Surface terrain',
                'land_ratio', 'total_surface', 'room_density', 'is_house',
                'is_large', 'is_small', 'month', 'season', 'Code departement',
                'commune_avg_price', 'dept_avg_price', 'commune_price_rank',
                'size_x_house', 'rooms_x_house', 'dept_squared', 'surface_squared'
            ]

            # Prepare train data
            input_feature_train_df = train_df[training_features]
            target_feature_train_df = train_df[TARGET_COLUMN]

            # Prepare test data
            input_feature_test_df = test_df[training_features]
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Fitting and transforming data")
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Concatenate transformed features with target
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            logging.info("Saving transformed data and preprocessor object")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            # Save preprocessor to final_model directory for inference
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            # Prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info("Data transformation completed successfully")
            return data_transformation_artifact

        except Exception as e:
            raise HousePredictionException(e, sys)