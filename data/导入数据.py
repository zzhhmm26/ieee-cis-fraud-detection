import kagglehub

# Download latest version
path = kagglehub.competition_download('ieee-fraud-detection')

print("Path to competition files:", path)