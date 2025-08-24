import pandas as pd

data = {'Color':['red', 'blue', 'green', 'blue', 'red']}
df = pd.DataFrame(data)

df_encoded = pd.get_dummies(df, columns=['Color'], prefix='color')

print("Original DataFrame:")
print(df)
print("\nEncoded DataFrame:")
print(df_encoded)    
