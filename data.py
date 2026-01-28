import kagglehub

# Download latest version
path = kagglehub.dataset_download("pranaykoppula/torgo-audio")

print("Path to dataset files:", path)
