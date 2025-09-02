import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import re
import config

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom transformer for text columns with comma-separated values"""
    
    def __init__(self, strategy='count'):
        self.strategy = strategy
        self.unique_values_ = {}
        
    def fit(self, X, y=None):
        for col in X.columns:
            if col in ['KronikHastalik', 'Alerji']:
                # Extract all unique conditions/allergies
                all_values = set()
                for value in X[col].dropna():
                    if isinstance(value, str):
                        if ',' in value:
                            all_values.update([v.strip() for v in value.split(',')])
                        else:
                            all_values.add(value.strip())
                self.unique_values_[col] = list(all_values)
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col in X_copy.columns:
            if col in self.unique_values_:
                if self.strategy == 'count':
                    # Create binary columns for each condition/allergy
                    for condition in self.unique_values_[col]:
                        col_name = f"{col}_{re.sub('[^a-zA-Z0-9]', '_', condition)}"
                        X_copy[col_name] = X_copy[col].apply(
                            lambda x: 1 if isinstance(x, str) and condition in [v.strip() for v in x.split(',')] else 0
                        )
                elif self.strategy == 'first':
                    # Keep only the first condition/allergy
                    X_copy[col] = X_copy[col].apply(
                        lambda x: x.split(',')[0].strip() if isinstance(x, str) and ',' in x else x
                    )
        
        return X_copy

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.preprocessor = None
        
    def handle_missing_values(self, missing_info):
        """Handle missing values based on the analysis"""
        
        # Drop columns with too many missing values
        high_missing_cols = missing_info[missing_info['Missing_Percentage'] > 
                                       config.MISSING_THRESHOLD * 100].index
        self.df = self.df.drop(columns=high_missing_cols)
        print(f"Dropped columns with >{config.MISSING_THRESHOLD*100}% missing values: {list(high_missing_cols)}")
        
        # Impute numerical variables
        numerical_cols = [col for col in config.NUMERICAL_COLS 
                         if col in self.df.columns and col != config.TARGET_COL]
        
        if numerical_cols:
            num_imputer = SimpleImputer(strategy='median')
            self.df[numerical_cols] = num_imputer.fit_transform(self.df[numerical_cols])
        
        # Impute categorical variables
        categorical_cols = [col for col in config.CATEGORICAL_COLS 
                           if col in self.df.columns]
        
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna('Unknown')
        
        # Handle target variable missing values
        if config.TARGET_COL in self.df.columns:
            self.df = self.df.dropna(subset=[config.TARGET_COL])
        
        return self.df
    
    def handle_outliers(self):
        """Handle outliers in numerical variables"""
        
        numerical_cols = [col for col in config.NUMERICAL_COLS 
                         if col in self.df.columns and col != config.TARGET_COL]
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
        
        return self.df
    
    def encode_categorical_variables(self):
        """Encode categorical variables"""
        
        # One-hot encoding for low cardinality features
        low_cardinality_cols = []
        high_cardinality_cols = []
        
        for col in config.CATEGORICAL_COLS:
            if col in self.df.columns:
                if self.df[col].nunique() <= 10:
                    low_cardinality_cols.append(col)
                else:
                    high_cardinality_cols.append(col)
        
        # One-hot encode low cardinality features
        if low_cardinality_cols:
            self.df = pd.get_dummies(self.df, columns=low_cardinality_cols, 
                                   drop_first=True)
        
        # Label encode high cardinality features (or consider frequency encoding)
        for col in high_cardinality_cols:
            if col in self.df.columns:
                # Frequency encoding
                freq_encoding = self.df[col].value_counts().to_dict()
                self.df[col] = self.df[col].map(freq_encoding)
                self.df[col] = self.df[col].fillna(0)  # For unseen categories
        
        return self.df
    
    def preprocess_text_columns(self):
        """Preprocess text columns with comma-separated values"""
        
        text_preprocessor = TextPreprocessor(strategy='count')
        text_cols = [col for col in config.TEXT_COLS if col in self.df.columns]
        
        if text_cols:
            text_features = text_preprocessor.fit_transform(self.df[text_cols])
            # Drop original text columns and add new features
            self.df = self.df.drop(columns=text_cols)
            self.df = pd.concat([self.df, text_features], axis=1)
        
        return self.df
    
    def scale_numerical_features(self):
        """Scale numerical features"""
        
        numerical_cols = [col for col in config.NUMERICAL_COLS 
                         if col in self.df.columns and col != config.TARGET_COL]
        
        if numerical_cols:
            scaler = StandardScaler()
            self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])
        
        return self.df
    
    def create_pipeline(self):
        """Create a preprocessing pipeline"""
        
        # Define column types for pipeline
        numerical_features = [col for col in config.NUMERICAL_COLS 
                            if col in self.df.columns and col != config.TARGET_COL]
        categorical_features = [col for col in config.CATEGORICAL_COLS 
                              if col in self.df.columns]
        
        # Create preprocessor
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self.preprocessor
    
    def run_complete_preprocessing(self, missing_info):
        """Run complete preprocessing pipeline"""
        
        print("Starting data preprocessing...")
        
        # Handle missing values
        self.handle_missing_values(missing_info)
        print("Missing values handled.")
        
        # Handle outliers
        self.handle_outliers()
        print("Outliers handled.")
        
        # Preprocess text columns
        self.preprocess_text_columns()
        print("Text columns processed.")
        
        # Encode categorical variables
        self.encode_categorical_variables()
        print("Categorical variables encoded.")
        
        # Scale numerical features
        self.scale_numerical_features()
        print("Numerical features scaled.")
        
        print(f"Final dataset shape: {self.df.shape}")
        print(f"Final columns: {list(self.df.columns)}")
        
        return self.df

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv(config.DATA_PATH)
    preprocessor = DataPreprocessor(df)
    
    # First run EDA to get missing info
    from eda_analysis import EDAnalysis
    eda = EDAnalysis(config.DATA_PATH)
    missing_info = eda.check_missing_values()
    
    # Preprocess data
    cleaned_df = preprocessor.run_complete_preprocessing(missing_info)
    cleaned_df.to_csv(config.OUTPUT_PATH, index=False)
    print("Cleaned data saved to:", config.OUTPUT_PATH)
