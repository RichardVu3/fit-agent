import os

# Define the directories
raw_dir = './results/raw'
results_dir = './results/results'

# Ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)

# Iterate over each file in the raw directory
for filename in os.listdir(raw_dir):
    # Create the new filename with the "result-" prefix
    new_filename = f"result-{filename}"
    # Define the full path for the new file
    new_file_path = os.path.join(results_dir, new_filename)
    # Create an empty file with the new name
    with open(new_file_path, 'w') as f:
        pass