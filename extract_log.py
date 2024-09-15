import re
import matplotlib.pyplot as plt

def extract_originial_sentences(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        
    # Define the pattern to match the general format with a random string of varying lengths
    pattern = r'\d{2}:\d{2}:\d{2}-\d{6} \d+\.\d+ [a-zA-Z_]+ .* \(cpu: \d+\.\d+ gpu: \d+\.\d+\) +run_mmlu\.py:\d+'
    
    # Find all matches in the content
    matches = re.findall(pattern, content)
    
    return matches

def save_sentences_to_file(sentences, output_file):
    with open(output_file, 'w') as file:
        for sentence in sentences:
            file.write(sentence + '\n')

def extract_sentences(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        
    # Define the pattern to match the general format with variations
    pattern = r'\d{2}:\d{2}:\d{2}-\d{6} (\d+\.\d+) .* \(cpu: (\d+\.\d+) gpu: (\d+\.\d+)\) +run_mmlu\.py:\d+'
    
    # Find all matches in the content
    matches = re.findall(pattern, content)
    # print(matches)
    
    lengths = []
    cpu_values = []
    gpu_values = []
    
    for match in matches:
        # print(match)
        lengths.append(float(match[0]))
        cpu_values.append(float(match[1]))
        gpu_values.append(float(match[2]))
    
    return lengths, cpu_values, gpu_values


def plot_data(lengths, cpu_values, gpu_values,output_file):
    cumulative_lengths = [sum(lengths[:i+1]) for i in range(len(lengths))]
    y_a = [cpu - cpu_values[0] for cpu in cpu_values]
    y_b = [gpu - gpu_values[0] for gpu in gpu_values]
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(cumulative_lengths, y_a, label='CPU memory increase')
    plt.plot(cumulative_lengths, y_b, label='GPU memory increase')
    
    plt.xlabel('Eval Time (s)')
    plt.ylabel('Memory increase (GB)')
    plt.title('MMLU benchmark of Llama-2-7B model')
    plt.legend()
    plt.grid(True)
    # plt.show()
    # Save the plot to a file
    plt.savefig(output_file)
    plt.close()


# Example usage
file_path = 'output.log'
output_file = 'output_plot.png'
output_log = 'result_data.log'
extracted_sentences = extract_originial_sentences(file_path)
for sentence in extracted_sentences:
    print(sentence)
lengths, cpu_values, gpu_values = extract_sentences(file_path)
save_sentences_to_file(extracted_sentences, output_log)


# for time in cpu_values:
#     print(time)
plot_data(lengths, cpu_values, gpu_values, output_file)