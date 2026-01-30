# Import data 
# Data imports from kaggle (https://www.kaggle.com/datasets/lameesmohammad/home-prices-in-canada)
import pandas as pd

file_path = "data/raw/HouseListings-Top45Cities-10292023-kaggle.csv"

df = pd.read_csv(
    file_path,
    encoding="latin1",      
    engine="python",        
    on_bad_lines="warn"     
)

print("Dataset shape:", df.shape)
display(df.head())