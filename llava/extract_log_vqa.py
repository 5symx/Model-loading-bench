import re
import matplotlib.pyplot as plt

def extract_originial_sentences(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        
    # Define the pattern to match the general format with a random string of varying lengths
    pattern = r'\d{2}:\d{2}:\d{2}-\d{6} \d+\.\d+ [a-zA-Z_]+ .* \(cpu: \d+\.\d+ gpu: \d+\.\d+\) +bench_vqa\.py:\d+'
    
    # Find all matches in the content
    matches = re.findall(pattern, content)
    
    return matches

def save_sentences_to_file(sentences, output_file):
    with open(output_file, 'w') as file:
        for sentence in sentences:
            file.write(sentence + '\n')

def extract_sentences(file_path):
    lengths = []
    cpu_values = []
    gpu_values = []
    state = []
    mapping = {
        "encoding": 0,
        "inference": 1,
        "decoding": 2
    }
    
    with open(file_path, 'r') as file:
        content = file.read()
    lines = content.split('\n')
    for line in lines:
        if 'totally take' in line:
            pattern = r'\d{2}:\d{2}:\d{2}-\d{6} \d+\.\d+ (\w+) : totally take (\d+\.\d+) second .* \(cpu: (\d+\.\d+) gpu: (\d+\.\d+)\) +bench_vqa\.py:\d+'
        elif 'bench_vqa' in line:
            pattern = r'(\w){2}:\d{2}:\d{2}-\d{6} (\d+\.\d+) .* \(cpu: (\d+\.\d+) gpu: (\d+\.\d+)\) +bench_vqa\.py:\d+'
        else:
            continue
        match = re.search(pattern, line)
        type_str = match.group(1)
        mapped_value = mapping.get(type_str, 3)  # Default to 3 if not found
        
        
        if 'totally take'  in line:
            print(match)
            print("Match at index % s, % s" % (match.start(), match.end())) 
            print("Full match: % s" % (match.group(0))) 
            print("state match: % s" % (mapped_value)) 
            print("state match: % s" % (match[2])) 
    # Define the pattern to match the general format with variations
    # pattern = r'\d{2}:\d{2}:\d{2}-\d{6} (\d+\.\d+) .* \(cpu: (\d+\.\d+) gpu: (\d+\.\d+)\) +run_mmlu\.py:\d+'
    
    # Find all matches in the content
    # matches = re.findall(pattern, content)
    # print(matches)
    
    
    # for match in matches:
    #     # print(match)
        
        state.append(float(mapped_value))
        lengths.append(float(match[2]))
        cpu_values.append(float(match[3]))
        gpu_values.append(float(match[4]))
    
    return lengths, cpu_values, gpu_values, state


def split_line(start_point, end_point,x_values):
    """
    Splits a line into three equal segments based on the x-axis values.

    Parameters:
    start_point (tuple): The starting point of the line (x1, y1).
    end_point (tuple): The ending point of the line (x2, y2).

    Returns:
    list: List of tuples representing the split points on the line.
    """
    x1, y1 = start_point
    x2, y2 = end_point
    
    # Calculate the increments for x and y

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    # Calculate the y-values for the given x-values
    y_values = [slope * x + intercept for x in x_values]
    points = list(zip(x_values, y_values))
    
    return points

def plot_data(lengths, cpu_values, gpu_values,output_file, state):
    # colors = {
    #     '0.0': 'blue',
    #     '1.0': 'red',
    #     '2.0': 'green',
    #     '3.0':'grey',
    # }
    colors = {
        '3.0': '#1f77b4',  # muted blue
        '2.0': '#9467bd',  # muted purple
        '1.0': '#2ca02c',  # cooked asparagus green
        '0.0': '#d62728',  # brick red
    }

    colors_2 = {
        '3.0': '#ff7f0e',  # safety orange
        '2.0': '#9467bd',  # muted purple
        '1.0': '#2ca02c',  # cooked asparagus green
        '0.0': '#d62728',  # brick red
    }


    import numpy as np
    # cumulative_lengths = np.cumsum([0] + lengths)
    cumulative_lengths = [sum(lengths[:i+1]) for i in range(len(lengths))]
    y_a = [cpu - cpu_values[0] for cpu in cpu_values]
    y_b = [gpu - gpu_values[0] for gpu in gpu_values]
    print(len(state) , len(cumulative_lengths))
    plt.figure(figsize=(10, 5))
    # for i in range(len(state)-1):
    i = 0
    while i < len(state)-1:
        print(state[i+1])
        if state[i+1] == 3.0:
            color = colors[str(state[i+1])]
            color_2 = colors_2[str(state[i+1])]
            plt.plot([cumulative_lengths[i], cumulative_lengths[i+1]], [y_a[i], y_a[i+1]], color=color, linewidth=2)
            plt.plot([cumulative_lengths[i], cumulative_lengths[i+1]], [y_b[i], y_b[i+1]], color=color_2, linewidth=2)
            i = i + 1
        else:
            x_values = [cumulative_lengths[i+1], cumulative_lengths[i+2],cumulative_lengths[i+3]]
            split_points = split_line((cumulative_lengths[i],y_a[i]),(cumulative_lengths[i+3],y_a[i+1]),x_values)
            plt.plot([cumulative_lengths[i], cumulative_lengths[i+1]], [y_a[i], split_points[0][1]], color=colors[str(state[i+1])], linewidth=2)
            plt.plot([cumulative_lengths[i+1], cumulative_lengths[i+2]], [split_points[0][1], split_points[1][1]], color=colors[str(state[i+2])], linewidth=2)
            plt.plot([cumulative_lengths[i+2], cumulative_lengths[i+3]], [split_points[1][1], y_a[i+1]], color=colors[str(state[i+3])], linewidth=2)
            split_points2 = split_line((cumulative_lengths[i],y_b[i]),(cumulative_lengths[i+3],y_b[i+1]),x_values)
            plt.plot([cumulative_lengths[i], cumulative_lengths[i+1]], [y_b[i], split_points2[0][1]], color=colors_2[str(state[i+1])], linewidth=2)
            plt.plot([cumulative_lengths[i+1], cumulative_lengths[i+2]], [split_points2[0][1], split_points2[1][1]], color=colors_2[str(state[i+2])], linewidth=2)
            plt.plot([cumulative_lengths[i+2], cumulative_lengths[i+3]], [split_points2[1][1], y_b[i+1]], color=colors_2[str(state[i+3])], linewidth=2)
            i = i + 3
        
        
    plt.text(61,16,'GPU memory usage',horizontalalignment='right')
    plt.text(61,2,'CPU memory usage',horizontalalignment='right')

    plt.text(25,25,'Model saving',horizontalalignment='right')
    plt.text(45,25,'Model loading',horizontalalignment='right')
    plt.text(55,25,'Benchmarking',horizontalalignment='right')
    
    plt.xlabel('Eval Time (s)')
    plt.ylabel('Memory usage (GB)')
    plt.title('MMLU benchmark of Llama-2-7B model')

    plt.grid(True)

    plt.savefig(output_file)
    plt.close()



# Example usage
file_path = 'output.log'
output_file = 'output_plot.png'
output_log = 'result_data.log'

extracted_sentences = extract_originial_sentences(file_path)
for sentence in extracted_sentences:
    print(sentence)
save_sentences_to_file(extracted_sentences, output_log)



lengths, cpu_values, gpu_values, state = extract_sentences(file_path)
plot_data(lengths, cpu_values, gpu_values, output_file, state)
# # for time in cpu_values:
# #     print(time)