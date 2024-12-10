import csv
import math
def get_duration(line):
    duration_value = line[2]
    if 'μs' in duration_value:
        time_value = float(duration_value.replace(' μs', '')) * 1e-6
    elif 'ns' in duration_value:
        time_value = float(duration_value.replace(' ns', '')) * 1e-9
    elif 'ms' in duration_value:
        time_value = float(duration_value.replace(' ms', '')) * 1e-3
    else:
        time_value = float(duration_value.replace('s', ''))
    return time_value

def avg_std(sums,flag):

        # Calculate the average between the two groups
    average_between_groups = sum(sums) / len(sums)

    # Calculate the standard deviation between the two groups
    variance = sum((x - average_between_groups) ** 2 for x in sums) / len(sums)
    std_deviation = math.sqrt(variance)

    # # Print the results
    # for i in range(len(sums)):
    print(f"length of {flag}: {len(sums)}")
    # print(f"{sums}")
    print(f"Average : {average_between_groups*1000:.3f} ms")
    print(f"Standard deviation : {std_deviation* 1000:.3f} ms")
    print()

# Function to calculate sum, average, and standard deviation between groups
def process_data(file_path):
    # Read data from CSV file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)[1:]
    
    # Initialize lists to store sums
    vmm_malloc_sums = []
    pure_malloc_sums = []
    vmm_copy_sums = []
    pure_copy_sums = []
    vmm_free_sums = []
    pure_free_sums = []
    # Process every four lines
    # for i in range(0, len(lines), 4):
    i = 0
    while i < len(lines):
        line = lines[i]
        if line[0] == 'cuMemCreate':
        # flag_value = float(line[0].replace('s', ''))
        
            # Extract the four lines
            group = lines[i:i+4]
            
            # Initialize sum for the current group
            group_sum = 0
            
            # Process each line in the group
            for line in group:
                # Extract the third value (duration in seconds)
                time_value = get_duration(line)
                group_sum += time_value
            
            # Store the sum
            vmm_malloc_sums.append(group_sum)
            i = i + 4
        elif line[0] == 'cudaMalloc':
            time_value = get_duration(line)
            pure_malloc_sums.append(time_value)
            i = i + 1

        elif line[0] == 'cudaMemcpy_vmm':
            time_value = get_duration(line)
            vmm_copy_sums.append(time_value)
            i = i + 1

        elif line[0] == 'cudaMemcpy_pure':
            time_value = get_duration(line)
            pure_copy_sums.append(time_value)
            i = i + 1
        
        elif line[0] == 'cuMemUnmap':
            time_value = get_duration(line)
            vmm_free_sums.append(time_value)
            i = i + 1

        elif line[0] == 'cudaFree':
            time_value = get_duration(line)
            pure_free_sums.append(time_value)
            i = i + 1
        else:
            print(f"ignore {line[0]}")
            print()
            i = i + 1
    
    vmm_malloc_chunks = [vmm_malloc_sums[i:i + 10] for i in range(0, len(vmm_malloc_sums), 10)]
    vmm_copy_chunks = [vmm_copy_sums[i:i + 10] for i in range(0, len(vmm_copy_sums), 10)]
    vmm_free_chunks = [vmm_free_sums[i:i + 10] for i in range(0, len(vmm_free_sums), 10)]
    for i in range(len(vmm_malloc_chunks)):
        print(f"round {i} ")
        avg_std(vmm_malloc_chunks[i],"vmm malloc")
        avg_std(vmm_copy_chunks[i],"vmm copy")
        avg_std(vmm_free_chunks[i],"vmm free")
    # return sums#, average_between_groups, std_deviation

    # print("frist round")
    # # print(vmm_free_sums)
    # avg_std(vmm_malloc_sums[:80], "vmm malloc")
    # # avg_std(pure_malloc_sums[:10]], "pure malloc")
    # avg_std(vmm_copy_sums[:80],"vmm copy")
    # # avg_std(pure_copy_sums[:10]],"pure copy")
    # avg_std(vmm_free_sums[:-1], "vmm free")
    # # avg_std(pure_free_sums[:10]], "pure free")

    # print("second round")
    # avg_std(vmm_malloc_sums[80:], "vmm malloc")
    # # avg_std(pure_malloc_sums[10:], "pure malloc")
    # avg_std(vmm_copy_sums[80:],"vmm copy")
    # # avg_std(pure_copy_sums[10:],"pure copy")
    # avg_std(vmm_free_sums[-1:], "vmm free")
    # # avg_std(pure_free_sums[10:], "pure free")


# Example usage
file_path = 'test_loop_all_10.csv'  # Replace with your CSV file path
process_data(file_path)

