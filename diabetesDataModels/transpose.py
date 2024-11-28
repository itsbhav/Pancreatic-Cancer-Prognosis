import pandas as pd

# Function to transpose a CSV file
def transpose_csv(input_file, output_file):
    # Read the CSV file
    data = pd.read_csv(input_file)
    
    # Transpose the data
    transposed_data = data.transpose()
    
    # Save the transposed data to a new CSV file
    transposed_data.to_csv(output_file, header=False)
    
    print(f"Transposed file saved to {output_file}")

# Example usage
input_csv = r"C:/Users/bhave/OneDrive/Documents/gene_expressions_GSE78721.csv"  # Replace with your input CSV file path
output_csv = "transposed_output_new.csv"  # Replace with desired output CSV file path
transpose_csv(input_csv, output_csv)
