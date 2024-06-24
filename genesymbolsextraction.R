# BiocManager::install("hgu133plus2.db")
library(hgu133plus2.db)

# List of gene IDs
# List of gene IDs
# gene_ids <- c("1553102_a_at", "200872_at", "201860_s_at", "202404_s_at",
#               "202504_at", "203021_at", "203108_at", "203476_at",
#               "203510_at", "203691_at", "203764_at", "203878_s_at",
#               "204351_at", "204653_at", "205780_at", "206026_s_at",
#               "206354_at", "209114_at", "209211_at", "209368_at",
#               "209955_s_at", "210495_x_at", "212236_x_at", "212353_at",
#               "212464_s_at", "213668_s_at", "216442_x_at", "219787_s_at",
#               "219901_at", "221900_at", "222449_at", "222810_s_at",
#               "223952_x_at", "228058_at", "230493_at", "238542_at",
#               "238689_at", "37892_at")
gene_ids <- c("1553102_a_at", "1555136_at", "1555137_a_at", "201417_at", "202267_at", "202286_s_at",
              "202504_at", "202935_s_at", "203476_at", "203510_at", "203757_s_at", "204320_at",
              "204351_at", "204602_at", "204713_s_at", "205941_s_at", "209016_s_at", "211719_x_at",
              "212354_at", "212444_at", "212464_s_at", "216442_x_at", "218856_at", "219901_at",
              "226237_at", "227051_at", "228923_at", "229479_at", "230831_at", "241137_at", "242397_at")

# Get gene annotations
gene_info <- select(hgu133plus2.db, keys = gene_ids, columns = c( "SYMBOL"), keytype = "PROBEID")

print(gene_info)

