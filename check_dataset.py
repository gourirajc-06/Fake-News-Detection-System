import pandas as pd

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print("="*50)
print("DATASET CHECK")
print("="*50)

print(f"\nFake.csv rows: {len(fake)}")
print(f"True.csv rows: {len(true)}")

# Check first few rows of Fake.csv
print("\n🔴 FAKE NEWS SAMPLE (First 2 rows):")
print(fake[['title', 'text']].head(2))

# Check first few rows of True.csv
print("\n🟢 REAL NEWS SAMPLE (First 2 rows):")
print(true[['title', 'text']].head(2))

# Check if labels are correct
print("\n✅ Label check after training:")
print("Fake news should have label = 0")
print("Real news should have label = 1")