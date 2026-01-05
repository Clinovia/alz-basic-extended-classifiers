import pickle

with open("dataset_sizes.pkl", "rb") as f:
    data = pickle.load(f)

# Inspect the contents
print(type(data))
print(data)
