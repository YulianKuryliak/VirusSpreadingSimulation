import pandas as pd

df = pd.read_csv (r'data.csv')
df1 = df[df['registration_area'] == 'Львівська']
print(df1.groupby('registration_region').sum())