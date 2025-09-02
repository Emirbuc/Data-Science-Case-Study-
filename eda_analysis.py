import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
import config

class EDAnalysis:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.setup_visualizations()
        
    def setup_visualizations(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def basic_info(self):
        print("="*50)
        print("DATASET BASIC INFORMATION")
        print("="*50)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes}")
        
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        
        print(f"\nDescriptive statistics:")
        print(self.df.describe())
        
    def check_missing_values(self):
        print("="*50)
        print("MISSING VALUES ANALYSIS")
        print("="*50)
        
        missing_info = pd.DataFrame({
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_info)
        
        # Visualize missing values
        plt.figure(figsize=(12, 6))
        msno.matrix(self.df)
        plt.title('Missing Values Matrix')
        plt.tight_layout()
        plt.savefig('missing_values_matrix.png')
        plt.close()
        
        return missing_info
    
    def analyze_target_variable(self):
        print("="*50)
        print("TARGET VARIABLE ANALYSIS: TedaviSuresi")
        print("="*50)
        
        target = self.df[config.TARGET_COL]
        
        print(f"Target variable statistics:")
        print(target.describe())
        
        # Check for outliers
        z_scores = np.abs(stats.zscore(target.dropna()))
        outliers = np.sum(z_scores > config.OUTLIER_THRESHOLD)
        print(f"\nNumber of outliers (z-score > {config.OUTLIER_THRESHOLD}): {outliers}")
        
        # Distribution plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(target, kde=True)
        plt.title('Distribution of Treatment Duration')
        plt.xlabel('Treatment Sessions')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(y=target)
        plt.title('Boxplot of Treatment Duration')
        
        plt.tight_layout()
        plt.savefig('target_distribution.png')
        plt.close()
    
    def analyze_categorical_variables(self):
        print("="*50)
        print("CATEGORICAL VARIABLES ANALYSIS")
        print("="*50)
        
        for col in config.CATEGORICAL_COLS:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(f"Unique values: {self.df[col].nunique()}")
                print(f"Value counts:\n{self.df[col].value_counts().head(10)}")
                
                # Plot top categories
                if self.df[col].nunique() <= 20:
                    plt.figure(figsize=(10, 6))
                    value_counts = self.df[col].value_counts()
                    sns.barplot(x=value_counts.values, y=value_counts.index)
                    plt.title(f'Distribution of {col}')
                    plt.tight_layout()
                    plt.savefig(f'categorical_{col}.png')
                    plt.close()
    
    def analyze_numerical_variables(self):
        print("="*50)
        print("NUMERICAL VARIABLES ANALYSIS")
        print("="*50)
        
        for col in config.NUMERICAL_COLS:
            if col in self.df.columns and col != config.TARGET_COL:
                print(f"\n{col}:")
                print(self.df[col].describe())
                
                # Distribution plot
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                sns.histplot(self.df[col], kde=True)
                plt.title(f'Distribution of {col}')
                
                plt.subplot(1, 2, 2)
                sns.boxplot(y=self.df[col])
                plt.title(f'Boxplot of {col}')
                
                plt.tight_layout()
                plt.savefig(f'numerical_{col}.png')
                plt.close()
    
    def analyze_correlations(self):
        print("="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Create numerical dataframe for correlation
        numerical_df = self.df[config.NUMERICAL_COLS].copy()
        
        # Calculate correlations
        correlation_matrix = numerical_df.corr()
        
        print("Correlation matrix:")
        print(correlation_matrix)
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        # Correlation with target
        target_corr = correlation_matrix[config.TARGET_COL].drop(config.TARGET_COL)
        print(f"\nCorrelation with {config.TARGET_COL}:")
        print(target_corr.sort_values(ascending=False))
    
    def analyze_text_columns(self):
        print("="*50)
        print("TEXT COLUMNS ANALYSIS")
        print("="*50)
        
        for col in config.TEXT_COLS:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(f"Number of unique entries: {self.df[col].nunique()}")
                print(f"Sample entries:\n{self.df[col].dropna().head(5)}")
                
                # Check for comma-separated values
                if col in ['KronikHastalik', 'Alerji']:
                    # Count occurrences of comma-separated values
                    comma_count = self.df[col].str.contains(',').sum()
                    print(f"Entries with multiple values (comma-separated): {comma_count}")
    
    def run_complete_analysis(self):
        """Run all EDA analyses"""
        self.basic_info()
        missing_info = self.check_missing_values()
        self.analyze_target_variable()
        self.analyze_categorical_variables()
        self.analyze_numerical_variables()
        self.analyze_correlations()
        self.analyze_text_columns()
        
        return missing_info

if __name__ == "__main__":
    eda = EDAnalysis(config.DATA_PATH)
    missing_info = eda.run_complete_analysis()
