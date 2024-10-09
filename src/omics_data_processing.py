# %% 
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (MONIKA)
project_dir = os.path.dirname(script_dir)
# Add the project directory to the Python path
sys.path.append(project_dir)
# Change the working directory to the project directory
os.chdir(project_dir)

import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
import glob
import zipfile
from tqdm import tqdm
import logging
from scipy import stats
import scipy.stats as stats
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import PowerTransformer
import statsmodels


# %% Dataset processing
# Load samples from CRC patients for both RNA (linked_rna.cct) and Protein (linked_rppa.tsv) data
linked_RNA = pd.read_csv('data/LinkedOmics/linked_rna.cct', sep='\t', index_col=0)
linked_rppa = pd.read_csv('data/LinkedOmics/linked_rppa.tsv', sep='\t', index_col=0)

# transpose
linked_RNA_T = linked_RNA.transpose()
linked_rppa_T = linked_rppa.transpose()

# Apply CMS labels obtained from the CMS classifier by Guinney et al. (2015)
classifier_labels = pd.read_csv('data/LinkedOmics/TCGACRC_CMS_CLASSIFIER_LABELS.tsv', sep='\t')
classifier_labels.rename(columns={classifier_labels.columns[0]: 'sample_ID'}, inplace=True)

# remove all columns except for 'sampleID' and 'SSP.nearestCMS'
classifier_labels = classifier_labels.loc[:, classifier_labels.columns.isin(['sample_ID', 'SSP.nearestCMS'])]

# set index to 'sample_ID'
classifier_labels.set_index('sample_ID', inplace=True)


# Merge the labels with the transposed RNA data
linked_RNA_T_labeled = linked_RNA_T.join(classifier_labels)
# Merge the labels with the transposed RPPA data
linked_rppa_T_labeled = linked_rppa_T.join(classifier_labels)


# only keep columns (genes) that are in both dataframes
linked_RNA_T_labeled = linked_RNA_T_labeled.loc[:, linked_RNA_T_labeled.columns.isin(linked_rppa_T_labeled.columns)]
linked_rppa_T_labeled = linked_rppa_T_labeled.loc[:, linked_rppa_T_labeled.columns.isin(linked_RNA_T_labeled.columns)]

# Due to data limitations, the proposed split is: CMS1 + CMS2 + CMS3 vs CMSALL (All subtypes)
# make subset df with only CMS1, CMS2, CMS3 and then drop the last column
linked_RNA_T_123 = linked_RNA_T_labeled.loc[linked_RNA_T_labeled.iloc[:, -1].isin(['CMS1', 'CMS2', 'CMS3'])].drop(columns=['SSP.nearestCMS'])
linked_rppa_T_123 = linked_rppa_T_labeled.loc[linked_rppa_T_labeled.iloc[:, -1].isin(['CMS1', 'CMS2', 'CMS3'])].drop(columns=['SSP.nearestCMS'])
linked_RNA_T_ALL = linked_RNA_T_labeled.loc[linked_RNA_T_labeled.iloc[:, -1].isin(['CMS1', 'CMS2', 'CMS3', 'CMS4'])].drop(columns=['SSP.nearestCMS'])
linked_rppa_T_ALL = linked_rppa_T_labeled.loc[linked_rppa_T_labeled.iloc[:, -1].isin(['CMS1', 'CMS2', 'CMS3', 'CMS4'])].drop(columns=['SSP.nearestCMS'])

# print shapes
print(f'linked_RNA_T_ALL shape: {linked_RNA_T_ALL.shape}')
print(f'linked_rppa_T_ALL shape: {linked_rppa_T_ALL.shape}')
print(f'linked_RNA_T_123 shape: {linked_RNA_T_123.shape}')
print(f'linked_rppa_T_123 shape: {linked_rppa_T_123.shape}')


# %% CHECK TO SEE IF EXPRESSION FROM GUINNEY DATA MATCHES THE RPPA DATA

def apply_yeo_johnson(column):
    """
    Applies a power transform to the data using the Yeo-Johnson method, to make it more Gaussian-like.
    """
    transformer = PowerTransformer(method='yeo-johnson')
    # Reshape data for transformation (needs to be 2D)
    column_reshaped = column.values.reshape(-1, 1)
    transformed_column = transformer.fit_transform(column_reshaped)
    # Flatten the array to 1D
    return transformed_column.flatten()

def filter_dataframes(df1, df2):
    # check overlap between df1.columns and df2.columns
    overlap = set(df1.columns).intersection(set(df2.columns))
    print('total number of variables in df2: {}'.format(len(df2.columns)))
    print(f'variables both in df1 and df2: {len(overlap)}')

    # keep only columns that are in overlap
    df1 = df1.loc[:, df1.columns.isin(overlap)]
    df2 = df2.loc[:, df2.columns.isin(overlap)]

    return df1, df2

def center_and_scale(df, axis=0):
    # center and scale across columns
    df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return df

# center and scale across columns
PROT_df_ALL_SC = center_and_scale(linked_rppa_T_ALL)
RNA_df_ALL_SC = center_and_scale(linked_RNA_T_ALL)
PROT_df_123_SC = center_and_scale(linked_rppa_T_123)
RNA_df_123_SC = center_and_scale(linked_RNA_T_123)



def transform_and_test(data_to_trim, dataframe_name):
    """
    Winsorize outliers, apply Yeo-Johnson transformation, and perform Kolmogorov-Smirnoff test to check for normality.
    """
    print('--------------------------------------------------------------')
    print(f'Initial results for {dataframe_name}')
    
    # Winsorize
    data_to_trim = data_to_trim.apply(lambda x: winsorize(x, limits=[0.01, 0.01]), axis=0)

    alpha = 0.05
    original_ks_results = {}

    # Perform initial K-S test and store results
    for column in data_to_trim.columns:
        data = data_to_trim[column].dropna()
        if data.nunique() > 1 and len(data) > 3:
            stat, p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            original_ks_results[column] = (stat, p)
            if p < alpha:
                # Apply Yeo-Johnson transformation
                data_to_trim[column] = apply_yeo_johnson(data_to_trim[column])

    # Perform K-S test again on transformed columns and compare
    for column, (original_stat, original_p) in original_ks_results.items():
        if original_p < alpha:
            transformed_data = data_to_trim[column].dropna()
            new_stat, new_p = stats.kstest(transformed_data, 'norm', args=(transformed_data.mean(), transformed_data.std()))
            if new_p < alpha:
                # Visualize and investigate columns (genes) that still deviate from normality
                print(f'Column: {column}')
                print(f'  Original K-S Statistic: {original_stat}, p-value: {original_p}')
                print(f'  Transformed K-S Statistic: {new_stat}, p-value: {new_p}')
                print('--------------------------------------------------------------\n')

                # make QQ-plot for these columns
                (fig, ax) = plt.subplots()
                stats.probplot(transformed_data, dist="norm", plot=ax)
                ax.set_title(f'QQ-plot for {column} (W: {round(new_stat, 3)}, p: {new_p}')
                # plt.show()
    
    return data_to_trim

# Apply transformations and tests to each dataframe
prot_df_all_transformed = transform_and_test(PROT_df_ALL_SC, 'PROT_df_ALL')
rna_df_all_transformed = transform_and_test(RNA_df_ALL_SC, 'RNA_df_ALL')
prot_df_123_transformed = transform_and_test(PROT_df_123_SC, 'PROT_df_123')
rna_df_123_transformed = transform_and_test(RNA_df_123_SC, 'RNA_df_123_SC')

print(prot_df_123_transformed.shape)

# whitelist genes that are relevant for CMS-based analysis
whitelist = ['VEGFR2', 'CDH1', 'BRAF', 'BAP1', 'TP53', 'CASP7', 'PRKCD', 'RAB11A', 'YAP1', 'CTNNB1', 'CCNB1', 'CCNE1', 
             'CCNE2', 'HSPA1A', 'ARID1A', 'ASNS', 'CHEK2', 'PCNA', 'ITGA2', 'MAPK1', 'ANXA1', 'CLDN7', 'COL6A1', 'FN1', 
             'MYH11','TP53BP1', 'EIF4EBP1', 'EEF2K', 'EIF4G1', 'FRAP1', 'RICTOR', 'RPS6', 'TSC1', 'RPS6KA1', 'ACACA',
             'AR', 'KIT', 'EGFR', 'FASN', 'ERBB3', 'IGFBP2', 'CDKN1A', 'CDKN1B', 'SQSTM1', 'PEA15', 'RB1', 'ACVRL1'
             'SMAD1', 'FOXM1', 'FOXO3', 'CAV1', 'PARK7', 'SERPINE1', 'RBM15', 'WWTR1', 'TGM2']
# blacklist genes that remain non-normal after transformation
blacklist = ['ERBB2', 'NKX2-1', 'RAD50']

# %%

# remove columns in blacklist
prot_df_all_transformed = prot_df_all_transformed.loc[:, ~prot_df_all_transformed.columns.isin(blacklist)]
rna_df_all_transformed = rna_df_all_transformed.loc[:, ~rna_df_all_transformed.columns.isin(blacklist)]
prot_df_123_transformed = prot_df_123_transformed.loc[:, ~prot_df_123_transformed.columns.isin(blacklist)]
rna_df_123_transformed = rna_df_123_transformed.loc[:, ~rna_df_123_transformed.columns.isin(blacklist)]

# check shape of all dataframes
print(f'\nexpr_df shape: {rna_df_all_transformed.shape}')
print(f'rppa_df shape: {prot_df_all_transformed.shape}')
print(f'expr_df_2_13 shape: {rna_df_123_transformed.shape}')
print(f'rppa_df_2_13 shape: {prot_df_123_transformed.shape}')

# Check for NaNs 
print(f'expr_df NaNs: {rna_df_all_transformed.isna().sum().sum()}')
print(f'rppa_df NaNs: {prot_df_all_transformed.isna().sum().sum()}')
# Check for Inf
print(f'expr_df Inf: {np.isinf(rna_df_all_transformed).sum().sum()}')
print(f'rppa_df Inf: {np.isinf(prot_df_all_transformed).sum().sum()}')

# replace NaNs with column mean
rna_df_all_transformed = rna_df_all_transformed.fillna(rna_df_all_transformed.mean())
prot_df_all_transformed = prot_df_all_transformed.fillna(prot_df_all_transformed.mean())
rna_df_123_transformed = rna_df_123_transformed.fillna(rna_df_123_transformed.mean())
prot_df_123_transformed = prot_df_123_transformed.fillna(prot_df_123_transformed.mean())


# write to csv
prot_df_all_transformed.to_csv('data/proteomics_for_pig_cmsALL.csv')
rna_df_all_transformed.to_csv('data/transcriptomics_for_pig_cmsALL.csv')

prot_df_123_transformed.to_csv('data/proteomics_for_pig_cms123.csv')
rna_df_123_transformed.to_csv('data/transcriptomics_for_pig_cms123.csv')

# write column names to .txt file
with open('data/VAR_NAMES_GENELIST.txt', 'w') as f:
    for item in prot_df_all_transformed.columns:
        f.write("%s\n" % item)
