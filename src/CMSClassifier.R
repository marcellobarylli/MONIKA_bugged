if (!require(devtools)) install.packages("devtools")
if (!require(CMSclassifier)) devtools::install_github("Sage-Bionetworks/CMSclassifier")

# if (!require("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
if (!require(org.Hs.eg.db)) BiocManager::install("org.Hs.eg.db")

# Load libraries
library(devtools)
library(CMSclassifier)
library(org.Hs.eg.db)

sampleData <- read.table("data/LinkedOmics/linked_rna.cct", header = T, row.names = 1) # replace with wherever your file is
# rownames(sampleData) <- sampleData$gene_name

# print head of rownames(sampleData)
head(rownames(sampleData))

entrez_ids <- mapIds(org.Hs.eg.db,
    keys = rownames(sampleData),
    column = "ENTREZID",
    keytype = "SYMBOL",
    multiVals = "first"
)
I <- !is.na(entrez_ids) # HUGO and Entrez are not a perfect match
sampleData <- sampleData[I, ]
rownames(sampleData) <- entrez_ids[I]
exp <- as.matrix(sampleData[, 2:ncol(sampleData)]) # first column is gene name
rownames(exp) <- rownames(sampleData)
colnames(exp) <- names(sampleData)[2:ncol(sampleData)]
classifyCMS.SSP(exp)

# Write to file
write.table(classifyCMS.SSP(exp), file = "data/LinkedOmics/TCGACRC_CMS_CLASSIFIER_LABELS.tsv", sep = "\t", quote = FALSE, col.names = NA)
