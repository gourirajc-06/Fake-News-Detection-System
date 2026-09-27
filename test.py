import pandas as pd

fake = pd.read_csv('dataset/Fake.csv')
true = pd.read_csv('dataset/True.csv')

print(f"Fake news count: {len(fake)}")
print(f"Real news count: {len(true)}")
print(fake.columns)  # ['title', 'text', 'subject', 'date']