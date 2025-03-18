import pandas as pd
import numpy as np
import csv

def convert_to_ms(duration):
    if 'ms' in duration:
        return float(duration.replace(' ms', ''))
    elif 'μs' in duration:
        return float(duration.replace(' μs', '')) / 1000
    else:
        return float(duration)


# Function to reformat data
def reformat_data(data):
    lines = data.strip().split('\n')
    headers = lines[0].replace('\t', ',').rstrip(',')
    formatted_lines = [headers]
    for line in lines[1:]:
        formatted_line = line.replace('\t', ',').rstrip(',')
        formatted_lines.append(formatted_line)
    return '\n'.join(formatted_lines)

# Function to read from a CSV file and write to another CSV file
def read_and_write_csv(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        data = '\n'.join([','.join(row) for row in reader])
    
    formatted_data = reformat_data(data)
    
    with open(output_file, mode='w', newline='') as outfile:
        outfile.write(formatted_data)

# Example usage
input_file = 'data_test.csv'
output_file = 'data_verify.csv'
read_and_write_csv(input_file, output_file)


# Read data from CSV file
data = pd.read_csv(output_file)
# print(data)

# # Convert data to DataFrame
# df = pd.DataFrame(data)

# Convert Duration to milliseconds
data['Duration'] = data['Duration'].apply(convert_to_ms)
data = data[5:]

# grouped_data = df.groupby('Name')['Duration'].agg(['mean', 'std']).reset_index()

# Group data by operation name and calculate average and standard deviation for each group
grouped_data = data.groupby('Name')['Duration'].agg(['mean', 'std']).reset_index()

# Rename columns for clarity
grouped_data.columns = ['Operation Name', 'Average Duration (ms)', 'Standard Deviation (ms)']

# Print the results in a table format
print(grouped_data)