"""
Configuration settings for the AI4I Predictive Maintenance Project.
"""
import logging
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Global Settings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 300 

RANDOM_STATE = 42

# 2. Directory Setup
ARTIFACTS_DIR = "artifacts"
sub_dirs = ["graphs", "models", "data"]
for d in sub_dirs:
    os.makedirs(os.path.join(ARTIFACTS_DIR, d), exist_ok=True)

# 3. Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(f"{ARTIFACTS_DIR}/pipeline_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 4. Data Constants
FILE_ID = "1lbc7JuoDQMTRengP2CVtnl-oohz-IE8x"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
FILEPATH = f"{ARTIFACTS_DIR}/data/ai4i2020.csv"

# 5. Modeling Constants
TARGET_COL = 'Machine failure'

# --- BUSINESS CONTEXT (CRITICAL FOR SENIOR ROLES) ---
# Assumption: A missed failure (machine blows up) costs $10k.
# Assumption: A false alarm (inspection time) costs $500.
COST_FALSE_NEGATIVE = 10000 
COST_FALSE_POSITIVE = 500    

# Columns to DROP (identifiers + leakage/outcomes)
LEAKAGE_COLS = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Features
CAT_FEATURES = ['Type']
NUM_FEATURES = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]', 
    'Temp_Diff',    # Engineered
    'Power',        # Engineered
    'Force_Ratio'   # Engineered
]