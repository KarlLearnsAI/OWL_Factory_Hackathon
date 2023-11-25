from pathlib import Path
import numpy as np
import pickle
import pandas as pd 




def add_time_features(df):
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    daytime_arr = []
    for idx in df.index:
        if df.loc[idx, "hour"] >= 6 and df.loc[idx, "hour"] <= 18:
            daytime_arr.append(1)
        else:
            daytime_arr.append(0)

    df['day_time'] = daytime_arr
    df['night_time'] = 1 - df['day_time']
    return df



filepath = Path(r"C:\Users\ethor\Desktop\hackathon\OWL_Factory_Hackathon\data\Eval_Y\2023_03_16_target.pq")

PATH_TO_DAY_MODEL = Path(r"C:\Users\ethor\Desktop\hackathon\OWL_Factory_Hackathon\submissions\Models\linreg.pkl")
PATH_TO_NIGHT_MODEL = Path(r"C:\Users\ethor\Desktop\hackathon\OWL_Factory_Hackathon\submissions\Models\linreg_2.pkl")



data = pd.read_parquet(filepath)

data = add_time_features(data)
data.index = pd.to_datetime(data.index)

day_data = data[data.day_time == 1]
night_data = data[data.day_time == 0]


with open(PATH_TO_DAY_MODEL, 'rb') as file:
    day_model = pickle.load(file)

with open(PATH_TO_NIGHT_MODEL, 'rb') as file:
    night_model = pickle.load(file)
    

day_data.iloc[:, 0] = day_model.predict(day_data.iloc[:,0])
night_data.iloc[:, 0] = night_model.predict(day_data.iloc[:,0])

data.iloc[:, 0] = list(day_data.iloc[:, 0].values) +  list(day_data.iloc[:, 0].values) 
data.index = list(day_data.index) + list(night_data.index)
print(data.index)




data.to_parquet()