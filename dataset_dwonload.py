import kagglehub

# Download latest version
path = kagglehub.dataset_download("apollo2506/eurosat-dataset")
print("Dataset downloaded to:", path)
