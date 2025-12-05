import time
import sys
import os

# Add the current directory to sys.path so we can import database
sys.path.append(os.getcwd())

print("Starting import of database...")
start_time = time.time()
try:
    import database
    end_time = time.time()
    print(f"Import took {end_time - start_time:.2f} seconds")
except Exception as e:
    print(f"Import failed: {e}")
