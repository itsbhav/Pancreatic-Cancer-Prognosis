# Load necessary library
library(dplyr)

# Read the original CSV file
df <- read.csv("C:/Users/bhave/OneDrive/Desktop/Minor/FinalDEG.csv")

# List of gene IDs you want to filter
# List of column names to include
columns_to_include <- feature_genes  # Add your column names here

# Select columns to include in the new dataframe
new_df <- df[, c(1, which(names(df) %in% columns_to_include))]

# Save the new dataframe as a CSV file
write.csv(new_df, "new_file1.csv", row.names = FALSE)
