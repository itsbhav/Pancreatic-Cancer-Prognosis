import pandas as pd

def filter_gene_symbols_with_data(input_csv, output_csv):
    # List of desired gene symbols
    desired_genes = [
        'CCDC69', 'FGD6', 'SOX4', 'LAMC2', 'TACSTD2', 'TRIM29', 'SOX9', 
        'TPBG', 'MET', 'CEACAM6', 'COL11A1', 'S100P', 'DKK1', 'F5', 
        'COL10A1', 'KRT7', 'FN1', 'SULF1', 'GPRC5A', 'TNFRSF21', 'COL8A1', 
        'BACE2', 'S100A6', 'LINC01614', 'FRMD5', 'MUCL3'
    ]

    # Read the input CSV file
    df = pd.read_csv(input_csv)
    print(df)
    df_new=pd.DataFrame()
    df_new['GeneSymbol']=df['GeneSymbol']
    # Assuming the first column is gene symbols and the first row contains column names
    for i in desired_genes:
        if i in df.columns:
            df_new[i]=df[i]
    print(df_new)
    # Filter the DataFrame to keep only the desired genes
    # filtered_df = df[df[gene_column].isin(desired_genes)].copy()

    # Save the filtered DataFrame to a new CSV file
    df_new.to_csv(output_csv, index=False)

    print(f"Filtered CSV saved to {output_csv}")
    # print(f"Number of genes retained: {len(filtered_df)}")
    print("Retained gene symbols:")
    # print(filtered_df[gene_column].tolist())

# Example usage
input_file = r'C:/Users/bhave/OneDrive/Desktop/Minor/transposed_output_new.csv'
output_file = 'filtered_genes_with_data_new.csv'
filter_gene_symbols_with_data(input_file, output_file)