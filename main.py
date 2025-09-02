import pandas as pd
from eda_analysis import EDAnalysis
from data_preprocessing import DataPreprocessor
import config

def main():
    print("Physical Medicine & Rehabilitation Data Analysis")
    print("="*50)
    
    try:
        # Step 1: EDA Analysis
        print("1. Performing Exploratory Data Analysis...")
        eda = EDAnalysis(config.DATA_PATH)
        missing_info = eda.run_complete_analysis()
        
        # Step 2: Data Preprocessing
        print("\n2. Performing Data Preprocessing...")
        df = pd.read_csv(config.DATA_PATH)
        preprocessor = DataPreprocessor(df)
        cleaned_df = preprocessor.run_complete_preprocessing(missing_info)
        
        # Step 3: Save cleaned data
        cleaned_df.to_csv(config.OUTPUT_PATH, index=False)
        print(f"\n3. Cleaned data saved to: {config.OUTPUT_PATH}")
        
        # Step 4: Final summary
        print("\n4. Analysis Complete!")
        print(f"Original data shape: {df.shape}")
        print(f"Cleaned data shape: {cleaned_df.shape}")
        print(f"Columns removed: {set(df.columns) - set(cleaned_df.columns)}")
        print(f"New columns added: {set(cleaned_df.columns) - set(df.columns)}")
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {config.DATA_PATH}")
        print("Please update the DATA_PATH in config.py")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
