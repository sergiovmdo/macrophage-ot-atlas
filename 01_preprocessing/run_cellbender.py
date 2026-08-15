import os
import csv
import subprocess
import sys
import glob

# --- 1. SET YOUR VARIABLES ---
DATASET_NAME = "bashore_run2"
BASE_DIR = "/storage/research/igmp_slide_workspace/GRP Zlobec/Sergio/MPI/Data"
TOTAL_DROPLETS_TO_INCLUDE = "20000" # Your agreed-upon value

# --- 2. SETUP TOP-LEVEL DIRECTORIES ---
SEARCH_DIR = os.path.join(BASE_DIR, DATASET_NAME)
# This is the parent directory that will hold all the sample folders
PARENT_OUTPUT_DIR = os.path.join(BASE_DIR, DATASET_NAME, "cellbender_output")
os.makedirs(PARENT_OUTPUT_DIR, exist_ok=True) # Ensure parent dir exists

# --- 3. FIND ALL SAMPLES ---
# Find all the raw .h5 files
search_pattern = os.path.join(SEARCH_DIR, "*/outs/raw_feature_bc_matrix.h5")
all_input_files = glob.glob(search_pattern)

if not all_input_files:
    print(f"Error: No 'raw_feature_bc_matrix.h5' files found in {SEARCH_DIR}")
    sys.exit(1)

print(f"Found {len(all_input_files)} samples to process.")
print("----------------------------------------------------\n")


# --- 4. START THE LOOP ---
for input_file in all_input_files:
    
    # --- 4a. Get paths for this specific sample ---
    outs_dir = os.path.dirname(input_file)
    sample_dir = os.path.dirname(outs_dir)
    sample_name = os.path.basename(sample_dir)
    
    # --- THIS IS THE FIX: Create a dedicated folder for this sample ---
    sample_output_dir = os.path.join(PARENT_OUTPUT_DIR, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Define all other paths
    metrics_file = os.path.join(outs_dir, "metrics_summary.csv")
    output_file = os.path.join(sample_output_dir, f"{sample_name}_corrected.h5")

    print(f"--- Processing Sample: {sample_name} ---")
    print(f"Input:   {input_file}")
    print(f"Output:  {output_file}")

    # Check if files exist
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file not found: {metrics_file}. Skipping this sample.")
        continue # Skip to the next file in the loop

    # --- 4b. Parse the CSV for this sample ---
    auto_expected_cells = ""
    try:
        with open(metrics_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            data = next(reader)
        
        try:
            col_index = header.index('Estimated Number of Cells')
        except ValueError:
            print(f'Error: Could not find "Estimated Number of Cells" in header {header}. Skipping.')
            continue
            
        value = data[col_index] # value is '4,680'
        auto_expected_cells = value.replace(',', '') # '4680'
        
        if not auto_expected_cells.isdigit():
            raise ValueError(f"Parsed value is not a digit: {auto_expected_cells}")
            
        print(f"Using Auto-Detected 'expected-cells': {auto_expected_cells}")

    except Exception as e:
        print(f"Error parsing CSV file {metrics_file}: {e}. Skipping this sample.")
        continue

    # --- 4c. Build the CellBender command ---
    command = [
        "cellbender", "remove-background",
        "--input", input_file,
        "--output", output_file,
        "--cuda",
        "--expected-cells", auto_expected_cells,
        "--total-droplets-included", TOTAL_DROPLETS_TO_INCLUDE,
        "--fpr", "0.01"
    ]

    print("\n--- Running CellBender Command ---")
    print(" ".join(command)) # Print the full command for review
    print("----------------------------------------------------\n")

    # --- 4d. Run with real-time verbosity ---
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end='')

        if proc.returncode != 0:
            print(f"\n--- Error: CellBender failed for {sample_name} with exit code {proc.returncode} ---")
        else:
            print(f"\n--- CellBender finished for {sample_name} ---")

    except FileNotFoundError:
        print("Error: 'cellbender' command not found. Is the 'cellbender' environment activated?")
        break # Exit the loop if cellbender isn't installed
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    print("\n----------------------------------------------------\n")

print("--- All samples processed ---")