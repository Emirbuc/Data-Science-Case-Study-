# Configuration file for data paths and settings
DATA_PATH = 'rehabilitation_data.csv'  # Update with your actual file path
OUTPUT_PATH = 'cleaned_data.csv'

# Column types
CATEGORICAL_COLS = ['Cinsiyet', 'KanGrubu', 'Uyruk', 'Bolum', 'Alerji', 'UygulamaYerleri']
TEXT_COLS = ['KronikHastalik', 'Tanilar', 'TedaviAdi']
NUMERICAL_COLS = ['Yas', 'TedaviSuresi', 'UygulamaSuresi']
ID_COL = 'HastaNo'
TARGET_COL = 'TedaviSuresi'

# Preprocessing parameters
MISSING_THRESHOLD = 0.3  # Threshold for dropping columns with too many missing values
OUTLIER_THRESHOLD = 3.0  # Z-score threshold for outlier detection
