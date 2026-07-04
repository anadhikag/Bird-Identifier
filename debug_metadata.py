import pickle

with open("vector_db/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print("Total records:", len(metadata))
print()

for i in range(5):
    print(metadata[i])
    print("-" * 60)