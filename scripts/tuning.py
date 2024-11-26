import subprocess
import re
import statistics

CMD = "./parallel_vlc"
CONFIG = "../dnn_wide_config.json"
NUM_CPU = 24
STEP = 2

def gen_search_space():
    search_space = []
    for i in range(9, NUM_CPU, STEP):
        for j in range(1, NUM_CPU, STEP):
            search_space.append((f"0-{i}", f"{NUM_CPU - 1 - j}-{NUM_CPU - 1}"))
    return search_space  

# Function to execute the command and capture output
def run_command(cpu1, cpu2):
    # Run the command and capture stdout and stderr
    result = subprocess.run([CMD, CONFIG, cpu1, cpu2], capture_output=True, text=True)
    
    # Combine stdout and stderr for easier processing
    output = result.stdout + result.stderr

    # Extract "Time" using regex
    # task1_time = re.search(r"Time:\s*([0-9.]+)\s*\(s\)", output)
    # task2_time = re.search(r"Compute time:\s*([0-9.]+)", output)
    task1_time = re.search(r"\ndnn runtime:\s*([0-9.]+)s", output)
    task2_time = re.search(r"\nwide dnn runtime:\s*([0-9.]+)s", output)
    
    if task1_time and task2_time:
        t1 = float(task1_time.group(1))
        t2 = float(task2_time.group(1))
        return t1, t2
    else:
        raise ValueError("Could not find 'Time' or 'Compute time' in the output")

# Main function to repeat the command 3 times and calculate median values
def main():
    search_space = gen_search_space()
    print(search_space, flush=True)
    # Repeat 3 times
    for cpu1, cpu2 in search_space:
        times = []
        max_retries = 3  # retry at most 3 times if enconter parsing issues
        attempt = 0

        while attempt < max_retries:
            try:
                for _ in range(3):
                    t1, t2 = run_command(cpu1, cpu2)
                    times.append((t1,t2))
                break  # Exit loop if successful
            except ValueError as e:
                attempt += 1
                times = []
                print(f"Found Error and Retry: {e}")
                if attempt == max_retries:
                    print("Max retries reached. Skip.")
    
        # Calculate median of both lists
        sums = [max(pair) for pair in times]
        median_time = statistics.median(sums)
        median_index = sums.index(median_time)

        # Report results
        print(f"{cpu1},{cpu2},{times[median_index][0]},{times[median_index][1]}", flush=True)

if __name__ == "__main__":
    main()