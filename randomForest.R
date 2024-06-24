# Load required packages
# install.packages('randomForest')
library(randomForest)
library(pheatmap)
library(ggplot2)
library(utils)

# Load data
train_data <- as.data.frame(read.csv("C:/Users/bhave/OneDrive/Desktop/Minor/FinalDEG.csv"))

# Convert 'Group' column to factor if needed
train_data$Group <- as.factor(train_data$Group)

# Extract gene expression data (excluding 'Id' and 'Group' columns)
gene_expression_data <- train_data[, !(names(train_data) %in% c("ID", "Group"))]

set.seed(41)
# Train a random forest model on the gene expression data for binary classification
rf_model <- randomForest(gene_expression_data, y = train_data$Group, ntree = 400, importance = TRUE, type = "classification")

# Get gene importance scores from the random forest model
gene_importance <- importance(rf_model)
# Filter feature genes based on importance score > 2
feature_genes <- rownames(gene_importance[gene_importance[, 1] > 2,])

# Print the identified feature genes
print(feature_genes)

# Subset the original data to include only feature genes
train_data_subset <- train_data[, c("Group", feature_genes)]

# Create a heatmap for visualization
heatmap_data <- as.matrix(train_data_subset[, -1])
rownames(heatmap_data) <- train_data_subset$Group

# Generate the heatmap using pheatmap
pheatmap(heatmap_data, clustering_distance_rows = "euclidean",
         clustering_distance_cols = "euclidean", fontsize = 8,
         display_numbers = TRUE, color = colorRampPalette(c("blue", "white", "red"))(50),
         main = "Feature Genes Heatmap")

# Optionally, you can save the heatmap plot to a file
# ggsave("feature_genes_heatmap.png", width = 8, height = 6)
