import pandas as pd
import unicodedata
df = pd.read_csv(r"C:\Users\HP\Documents\Master's thesis\collected data\collected data.csv")
# print(df)
df['state'] = df['state'].replace({'successful': 1,'failed':0})
count = df['state'].value_counts()
count_countries = df['country'].value_counts()
print(count)
print(count_countries)

