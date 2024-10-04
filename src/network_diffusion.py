# %%
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import math
import random
import copy
import time
from mpi4py import MPI
import pickle as pkl
import os
from tqdm import tqdm
import argparse
import sys
import csv

if not"SLURM_JOB_ID" in os.environ:
    # Figure export settings
    from mpl_toolkits.axes_grid1 import ImageGrid 
    plt.rcParams.update(plt.rcParamsDefault) 
    plt.rcParams.update({"font.size": 15,
                        "figure.dpi" : 100,
                        "grid.alpha": 0.3,
                        "axes.grid": True,
                        "axes.axisbelow": True, 
                        "figure.figsize": (8,6), 
                        "mathtext.fontset":"cm",
                        "xtick.labelsize": 14, 
                        "ytick.labelsize": 14, 
                        "axes.labelsize": 16, 
                        "legend.fontsize": 13.5})
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")



if not "SLURM_JOB_ID" in os.environ:
    import pymnet as pn

# Check if the script is running in an environment with predefined sys.argv (like Jupyter or certain HPC environments)
if 'ipykernel_launcher.py' in sys.argv[0] or 'mpirun' in sys.argv[0]:
    # Create a list to hold the arguments you want to parse
    args_to_parse = []

    # Iterate through the system arguments
    for arg in sys.argv:
        # Add only your specified arguments to args_to_parse
        if '--koh' in arg or '--kob' in arg or '--cms' in arg:
            args_to_parse.extend(arg.split('='))
else:
    args_to_parse = sys.argv[1:]  # Exclude the script name

# Command Line Arguments
parser = argparse.ArgumentParser(description='Run QJ Sweeper with command-line arguments.')
parser.add_argument('--koh', type=int, default=0, help='Number of hub nodes to knock out')
parser.add_argument('--kob', type=int, default=0, help='Number of bottom nodes to knock out')
parser.add_argument('--red_range', type=str, default='0.05,0.05,1', help='Range of reduction factors to investigate')
parser.add_argument('--cms', type=str, default='cmsALL', choices=['cmsALL', 'cms123'], help='CMS to use')
# parser.add_argument('--mode', type=str, default='disruption', choices=['disruption', 'transition'], help='Type of knockout analysis')
parser.add_argument('--test_net', type=bool, default=False, help='Boolean for using test nets of various sizes')
parser.add_argument('--pathway', type=bool, default=False, help='Boolean for Pathway Knockout')
parser.add_argument('--path_size_range', type=str, default='5,26', help='Range of reduction factors to investigate for permutation')
parser.add_argument('--permu_runs', type=int, default=None, help='Number of runs for permutation random pathway knockout')
parser.add_argument('--visualize', type=bool, default=False, help='Boolean for visualizing the network')
parser.add_argument('--enrich_file', type=str, default='/home/mbarylli/thesis_code/Diffusion/data_for_diffusion/Pathway_Enrichment_Info_LinkedOmics.csv', help='Path to pathway enrichment file')
parser.add_argument('--symmetric', type=bool, default=True, help='Boolean for symmetric knockdown')
parser.add_argument('--net_dens', type=str, default='low_dens', help='Network density')

args = parser.parse_args(args_to_parse)

# Get range of expression reduction values from parsed args
red_range = args.red_range.split(',')
red_range = np.linspace(float(red_range[0]), float(red_range[1]), int(float(red_range[2])))

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if "SLURM_JOB_ID" not in os.environ:
    rank = 0
    size = 1


# %% FUNCTION TO CREATE MULTI-OMICS GRAPH
def weighted_multi_omics_graph(cms, verbose=False):
    """
    Creates a 2-layer multiplex network, using the adjacency matrices of the proteomics and transcriptomics data obtained during the network inference stage.
    params:
        cms: 'cms123' or 'cmsALL', to specify the aggressive (cmsALL) or non-aggressive (cms123) CMS subtypes
        plot: Boolean to plot the degree distribution
        verbose: Boolean to print the orphans
    returns:
        weighted_G_multiplex: Multiplex graph with weighted edges
        M: Pymnet multiplex network
    """
    p = 154 
    cms = cms
    man = False

    if "SLURM_JOB_ID" in os.environ:
        adj_matrix_proteomics = pd.read_csv(f'/home/mbarylli/thesis_code/Diffusion/data_for_diffusion/inferred_adjacencies/proteomics_{cms}_adj_matrix_p{p}_Lambda_np{not man}_{args.net_dens}.csv', index_col=0)
        adj_matrix_transcriptomics = pd.read_csv(f'/home/mbarylli/thesis_code/Diffusion/data_for_diffusion/inferred_adjacencies/transcriptomics_{cms}_adj_matrix_p{p}_Lambda_np{not man}_{args.net_dens}.csv', index_col=0)
    else: 
        adj_matrix_proteomics = pd.read_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/MONIKA/Networks/net_results/inferred_adjacencies/proteomics_{cms}_adj_matrix_p{p}_Lambda_np{not man}_{args.net_dens}.csv', index_col=0)
        adj_matrix_transcriptomics = pd.read_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/MONIKA/Networks/net_results/inferred_adjacencies/transcriptomics_{cms}_adj_matrix_p{p}_Lambda_np{not man}_{args.net_dens}.csv', index_col=0)


    # Create separate graphs for each adjacency matrix
    G_proteomics_layer = nx.from_pandas_adjacency(adj_matrix_proteomics)
    G_transcriptomic_layer = nx.from_pandas_adjacency(adj_matrix_transcriptomics)

    # # get orphans using function
    orphans_proteomics = get_orphans(G_proteomics_layer)
    orphans_transcriptomics = get_orphans(G_transcriptomic_layer)

    if verbose == True:
        print(f'orphans in proteomics: {orphans_proteomics}')
        print(f'orphans in transcriptomics: {orphans_transcriptomics}')

    # Function to add a suffix to node names based on layer
    def add_layer_suffix(graph, suffix):
        return nx.relabel_nodes(graph, {node: f"{node}{suffix}" for node in graph.nodes})

    # Create separate graphs for each adjacency matrix and add layer suffix
    G_proteomics_layer = add_layer_suffix(nx.from_pandas_adjacency(adj_matrix_proteomics), '.p')
    G_transcriptomic_layer = add_layer_suffix(nx.from_pandas_adjacency(adj_matrix_transcriptomics), '.t')

    # Create a multiplex graph
    G_multiplex = nx.Graph()

    # Add nodes and edges from both layers
    G_multiplex.add_nodes_from(G_proteomics_layer.nodes(data=True), layer='PROTEIN')
    G_multiplex.add_edges_from(G_proteomics_layer.edges(data=True), layer='PROTEIN')
    G_multiplex.add_nodes_from(G_transcriptomic_layer.nodes(data=True), layer='RNA')
    G_multiplex.add_edges_from(G_transcriptomic_layer.edges(data=True), layer='RNA')

    common_nodes = set(adj_matrix_proteomics.index).intersection(adj_matrix_transcriptomics.index)

    inter_layer_weight = 1
    # Add nodes and edges from both layers
    G_multiplex.add_nodes_from(G_proteomics_layer.nodes(data=True), layer='PROTEIN')
    G_multiplex.add_edges_from(G_proteomics_layer.edges(data=True), layer='PROTEIN')
    G_multiplex.add_nodes_from(G_transcriptomic_layer.nodes(data=True), layer='RNA')
    G_multiplex.add_edges_from(G_transcriptomic_layer.edges(data=True), layer='RNA')

    common_nodes = set(adj_matrix_proteomics.index).intersection(adj_matrix_transcriptomics.index)

    inter_layer_weight = 1
    # Add inter-layer edges for common nodes
    for node in common_nodes:
        G_multiplex.add_edge(f"{node}.p", f"{node}.t",layer='interlayer')

    weighted_G_multiplex = G_multiplex.copy()
    for u, v, data in weighted_G_multiplex.edges(data=True):
        if data.get('layer') == 'interlayer':
            weighted_G_multiplex[u][v]['weight'] = inter_layer_weight
        else:
            weighted_G_multiplex[u][v]['weight'] = 1.0
    for u, v, data in weighted_G_multiplex.edges(data=True):
        if data.get('layer') == 'interlayer':
            weighted_G_multiplex[u][v]['weight'] = inter_layer_weight
        else:
            weighted_G_multiplex[u][v]['weight'] = 1.0


    # PYMNET (For fancy multiplex visualisation)
    if not "SLURM_JOB_ID" in os.environ:
        # Initialize a pymnet multilayer network
        M = pn.MultiplexNetwork(couplings=('categorical', 1), fullyInterconnected=False)

        def preprocess_node_name(node_name):
            # If node ends with '.p' or '.t', remove the suffix
            if node_name.endswith('.p') or node_name.endswith('.t'):
                return node_name[:-2]  # Assuming suffixes are always two characters
            return node_name

        # Add nodes and edges for proteomics layer
        for node in G_proteomics_layer.nodes:
            # Preprocess node names to remove suffixes
            processed_node = preprocess_node_name(node)
            M.add_node(processed_node, layer='PROTEIN')
        for u, v in G_proteomics_layer.edges:
            # Preprocess node names for each edge
            processed_u = preprocess_node_name(u)
            processed_v = preprocess_node_name(v)
            M[processed_u, processed_v, 'PROTEIN', 'PROTEIN'] = 1

        # Add nodes and edges for transcriptomic layer
        for node in G_transcriptomic_layer.nodes:
            # Preprocess node names to remove suffixes
            processed_node = preprocess_node_name(node)
            M.add_node(processed_node, layer='RNA')
        for u, v in G_transcriptomic_layer.edges:
            # Preprocess node names for each edge
            processed_u = preprocess_node_name(u)
            processed_v = preprocess_node_name(v)
            M[processed_u, processed_v, 'RNA', 'RNA'] = 1

        return weighted_G_multiplex, M  #, rna_node_positions
    else:
        return weighted_G_multiplex, None       

# get number of orphans
def get_orphans(G):
    orphans = []
    for node in G.nodes():
        if G.degree(node) == 0:
            orphans.append(node)
    return orphans


weighted_G_cms_123, pymnet_123 = weighted_multi_omics_graph('cms123')
weighted_G_cms_ALL, pymnet_ALL = weighted_multi_omics_graph('cmsALL')


orphans_123 = get_orphans(weighted_G_cms_123)
orphans_ALL = get_orphans(weighted_G_cms_ALL)

# Since the RNA and prot layers have different orphans, the multiplex should have no orphans
if orphans_123 or orphans_ALL:
    print(f'Multiplex orphans in cms123: {orphans_123}')  
    print(f'Multiplex orphans in cmsALL: {orphans_ALL}')


# %%

# ALTERNATIVE to multi-omics graph, small synthetic graph can be generated. Good for quick testing. 
def create_multiplex_test(num_nodes, inter_layer_weight=1.0):
    """
    Creates a multiplex graph with two layers, each having the specified number of nodes.
    params:
        num_nodes: Number of nodes in each layer
        inter_layer_weight: Weight of the inter-layer edges
    returns:
        G: Multiplex NetworkX graph
    """
    G = nx.Graph()

    # Define nodes for each layer
    nodes_layer_p = [f"{i}.p" for i in range(num_nodes)]
    nodes_layer_t = [f"{i}.t" for i in range(num_nodes)]

    # Add nodes
    G.add_nodes_from(nodes_layer_p)
    G.add_nodes_from(nodes_layer_t)

    # Add random edges within each layer
    for _ in range(num_nodes * 2):  # Randomly adding double the number of nodes as edges in each layer
        u, v = np.random.choice(nodes_layer_p, 2, replace=False)
        G.add_edge(u, v, weight=1.0)

        u, v = np.random.choice(nodes_layer_t, 2, replace=False)
        G.add_edge(u, v, weight=1.0)

    # Add edges between corresponding nodes of different layers
    for i in range(num_nodes):
        G.add_edge(nodes_layer_p[i], nodes_layer_t[i], weight=inter_layer_weight)

    return G

# %% ##### FUNCTION DEFINITIONS FOR GRAPH LAPLACIANS AND KNOCKOUTS ##########
# Adjust Laplacian Matrix Calculation for Weighted Graph
def weighted_laplacian_matrix(G):
    """
    Calculate the Laplacian matrix for a weighted graph.
    params:
        G: graph in nx format
    returns:
        L: Laplacian matrix
    """
    node_order = list(G.nodes())
    # Weighted adjacency matrix
    W = nx.to_numpy_array(G, nodelist=node_order, weight='weight')
    # Diagonal matrix of vertex strengths
    D = np.diag(W.sum(axis=1))
    # Weighted Laplacian matrix
    L = D - W

    return L

def force_dense_eig_solver(L):
    """
    Forcing the use of a dense eigensolver for the Laplacian matrix, rather than sparse methods.
    params: 
        L: Laplacian matrix
    returns:
        eigenvalues: Eigenvalues of the Laplacian matrix
        eigenvectors: Eigenvectors of the Laplacian matrix
    """
    if issparse(L):
        L = L.toarray()  # Convert sparse matrix to dense
    eigenvalues, eigenvectors = np.linalg.eigh(L)  # Use dense method

    return eigenvalues, eigenvectors

def laplacian_exponential_kernel_eigendecomp(L, t):
    """
    Function to compute the Laplacian exponential diffusion kernel using eigen-decomposition
    The function takes the Laplacian matrix L and a time parameter t as inputs.
    params:
        L: Laplacian
        t: timestep
    returns:
        kernel: Laplacian exponential diffusion kernel
    """
    # Calculate the eigenvalues and eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = force_dense_eig_solver(L)
    # Compute the matrix exponential using eigenvalues
    exp_eigenvalues = np.exp(-t * eigenvalues)
    # Reconstruct the matrix using the eigenvectors and the exponentiated eigenvalues
    kernel = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T

    return kernel

def laplacian_eigendecomp(L):
    """
    Function to eigen-decomposition of the Laplacian matrix, 
    params:
        L: Laplacian
    returns: 
        eigenvalues: Eigenvalues of the Laplacian matrix
        eigenvectors: Eigenvectors of the Laplacian matrix
    """
    # Calculate the eigenvalues and eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = force_dense_eig_solver(L)

    return eigenvalues, eigenvectors

def laplacian_exponential_diffusion_kernel(L, t):
    """
    compute the Laplacian exponential kernel for a given t value"""
    
    return scipy.linalg.expm(-t * L)

def knockout_node(G, node_to_isolate):
    """
    Isolates a node in the graph by removing all edges connected to it.
    
    params
        G: NetworkX graph
        node_to_isolate: Node to isolate
    return: 
        modified_graph: Graph with the node isolated
        new_laplacian: Laplacian matrix of the modified graph
    """
    modified_graph = G.copy()
    # Remove all edges to and from this node
    edges_to_remove = list(G.edges(node_to_isolate))
    modified_graph.remove_edges_from(edges_to_remove)
    new_laplacian = weighted_laplacian_matrix(modified_graph)

    return modified_graph, new_laplacian

def knockdown_node_both_layers(G, node_to_isolate_base, reduced_weight=0.3):
    """
    Reduces the weights of all edges connected to a node in both layers of the graph.
    params:
        G: NetworkX graph
        node_to_isolate_base: Base node name whose edges will be reduced in both layers
        reduced_weight: Factor to reduce edge weights by, defaults to 0.5
    returns: 
        Tuple containing the modified graph and its weighted Laplacian matrix
    """

    modified_graph = G.copy()
    
    # Add layer suffixes to the base node name
    node_to_isolate_proteomics = f"{node_to_isolate_base}.p"
    node_to_isolate_transcriptomics = f"{node_to_isolate_base}.t"
    
    # Reduce the weight of all edges to and from this node in both layers
    for node_to_isolate in [node_to_isolate_proteomics, node_to_isolate_transcriptomics]:
        for neighbor in G[node_to_isolate]:
            modified_graph[node_to_isolate][neighbor]['weight'] = reduced_weight
            modified_graph[neighbor][node_to_isolate]['weight'] = reduced_weight
    
    # Compute the weighted Laplacian matrix for the modified graph
    new_laplacian = weighted_laplacian_matrix(modified_graph)

    return modified_graph, new_laplacian, 1, 1, 1

def knockdown_node_single_layer(G, node_to_isolate_base, layer_suffix, reduced_weight=0.3):
    """
    Reduces the weights of all edges connected to a node in one specified layer of the graph.
    Initially sets all edge weights to 1 if not already set.

    params:
        G: NetworkX graph
        node_to_isolate_base: Base node name whose edges will be reduced
        layer_suffix: Suffix indicating the layer ('.p' for proteomics, '.t' for transcriptomics)
        reduced_weight: Factor to reduce edge weights by, defaults to 0.3
    :returns
        Tuple containing the modified graph and its weighted Laplacian matrix
    """
    if layer_suffix not in ['.p', '.t']:
        raise ValueError("Invalid layer suffix. Choose '.p' for proteomics or '.t' for transcriptomics.")

    modified_graph = G.copy()

    # Initially set all edge weights to 1
    for u, v in modified_graph.edges():
        modified_graph[u][v]['weight'] = 1

    # Add layer suffix to the base node name
    node_to_isolate = f"{node_to_isolate_base}{layer_suffix}"
    
    # Check if the node exists in the graph
    if node_to_isolate not in G:
        raise ValueError(f"Node {node_to_isolate} does not exist in the graph.")

    # Reduce the weight of all edges to and from this node in the specified layer
    for neighbor in G[node_to_isolate]:
        modified_graph[node_to_isolate][neighbor]['weight'] = reduced_weight
        modified_graph[neighbor][node_to_isolate]['weight'] = reduced_weight

    # Compute the weighted Laplacian matrix for the modified graph
    new_laplacian = weighted_laplacian_matrix(modified_graph)

    return modified_graph, new_laplacian


def knockdown_pathway_nodes(G, pathway_description, reduced_weight=0.3):
    """
    Reduces the weights of all edges connected to the nodes in a pathway in both layers of the graph.
    params:
        G: NetworkX graph
        pathway_description: Description of the pathway whose nodes will be reduced in both layers
        reduced_weight: Factor to reduce edge weights by, defaults to 0.3
    returns:
        Tuple containing the modified graph and its weighted Laplacian matrix
        num_disconnected_components: Number of disconnected components in the subgraph
        connected_components_lengths: Lengths of connected components in the subgraph
        len(pathway_nodes_with_suffixes): Number of nodes in the pathway
    """

    # Find rows where 'Pathway' column contains the given string
    rows = pathway_df[pathway_df['Pathway'] == pathway_description]
    
    # Initialize a list to store the base node names
    base_node_names = []

    # Iterate over the found rows
    for _, row in rows.iterrows():
        # Split the 'genes' column into individual genes and add them to the list
        base_node_names.extend(row['genes'].split('|'))

    modified_graph = G.copy()
    
    # Iterate over the base node names
    for node_to_isolate_base in base_node_names:
        # Add layer suffixes to the base node name
        node_to_isolate_proteomics = f"{node_to_isolate_base}.p"
        node_to_isolate_transcriptomics = f"{node_to_isolate_base}.t"
        
        # Reduce the weight of all edges to and from this node in both layers
        for node_to_isolate in [node_to_isolate_proteomics, node_to_isolate_transcriptomics]:
            if node_to_isolate in G:
                for neighbor in G[node_to_isolate]:
                    modified_graph[node_to_isolate][neighbor]['weight'] = reduced_weight
                    modified_graph[neighbor][node_to_isolate]['weight'] = reduced_weight
    
    # Compute the weighted Laplacian matrix for the modified graph
    new_laplacian = weighted_laplacian_matrix(modified_graph)

    ### COMNNECTED COMPONENTS
    pathway_nodes_with_suffixes = [f"{node}.p" for node in base_node_names] + [f"{node}.t" for node in base_node_names]
    subgraph = modified_graph.subgraph(pathway_nodes_with_suffixes)
    connected_components = list(nx.connected_components(subgraph))
    # get lengths of connected components
    connected_components_lengths = [len(i) for i in connected_components]
    num_disconnected_components = len(connected_components)
    ###


    return modified_graph, new_laplacian, num_disconnected_components, connected_components_lengths, len(pathway_nodes_with_suffixes)


def knockdown_random_nodes(G, node_list, reduced_weight=0.05):
    """
    Reduces the weights of all edges connected to the nodes in a pathway or a list of nodes in both layers of the graph.
    params:
        G: NetworkX graph
        num_nodes: Number of nodes to randomly select for knockdown
        reduced_weight: Factor to reduce edge weights by, defaults to 0.3
    returns:
        Tuple containing the modified graph and its weighted Laplacian matrix
        num_disconnected_components: Number of disconnected components in the subgraph
        connected_components_lengths: Lengths of connected components in the subgraph
        len(pathway_nodes_with_suffixes): Number of nodes in the pathway
    """


    modified_graph = G.copy()
    
    # Reduce the weight of all edges to and from the random nodes in both layers
    for node in node_list:
        if node in G:  # Check if the node is present in the graph
            for neighbor in G[node]:  # Iterate through its neighbors
                # Reducing the weights
                modified_graph[node][neighbor]['weight'] = reduced_weight
                modified_graph[neighbor][node]['weight'] = reduced_weight
        else:
            print(f"Node {node} not found in graph!!!!!!!!!!!")

    # Form the subgraph for the randomly selected nodes
    subgraph = modified_graph.subgraph(node_list)
    # Find the connected components in the subgraph
    connected_components = list(nx.connected_components(subgraph))
    connected_components_lengths = [len(i) for i in connected_components]

    # Calculate the number of disconnected components (which is the length of connected components list)
    num_disconnected_components = len(connected_components)
    # Compute the weighted Laplacian matrix for the modified graph
    new_laplacian = nx.laplacian_matrix(modified_graph)

    # Return the modified graph, new laplacian, and number of disconnected components
    return modified_graph, new_laplacian, num_disconnected_components, connected_components_lengths, len(node_list)

# %% #################### MPI PARALLELIZATION & RUN FUNCTION DEFINITIONS ####################

# PARALLEL PROCESSING: Function to distribute nodes across ranks
def distribute_nodes(nodes, rank, size):
    """
    Distributes nodes across ranks.
    params:
        nodes: List of nodes
        rank: Rank of the current process
        size: Total number of processes
    returns:
        List of nodes assigned to the current rank
    """
    num_nodes = len(nodes)
    nodes_per_proc = num_nodes // size
    remainder = num_nodes % size

    if rank < remainder:
        start_index = rank * (nodes_per_proc + 1)
        end_index = start_index + nodes_per_proc + 1
    else:
        start_index = remainder * (nodes_per_proc + 1) + (rank - remainder) * nodes_per_proc
        end_index = start_index + nodes_per_proc

    return nodes[start_index:end_index]


def distribute_pathways(pathways, rank, size):
    """
    Distributes pathways across ranks.
    params:
        pathways: List of pathways
        rank: Rank of the current process
        size: Total number of processes
    returns:
        List of pathways assigned to the current rank
    """
    num_pathways = len(pathways)
    pathways_per_proc = num_pathways // size
    remainder = num_pathways % size

    if rank < remainder:
        start_index = rank * (pathways_per_proc + 1)
        end_index = start_index + pathways_per_proc + 1
    else:
        start_index = remainder * (pathways_per_proc + 1) + (rank - remainder) * pathways_per_proc
        end_index = start_index + pathways_per_proc

    return pathways[start_index:end_index]

def distribute_runs(total_runs, rank, size):
    """
    Distribute runs across ranks.
    """
    runs_per_rank = total_runs // size
    start_run = rank * runs_per_rank
    end_run = start_run + runs_per_rank if rank != size - 1 else total_runs  # Ensure the last rank takes any remaining runs
    return range(start_run, end_run)


def generate_node_combinations(G, num_nodes, total_combinations=10000):
    """
    Generates unique combinations of nodes for knockout analysis. For generating random null distribution of knockouts.
    params:
        G: NetworkX graph
        num_nodes: Number of nodes to select
        total_combinations: Total number of combinations to generate
    returns:
        List of unique combinations of nodes
    """
    local_random = random.Random(42)  # using a local instance of Random

    # Separate nodes by layer
    proteomics_nodes = [node for node in G.nodes() if node.endswith('.p')]

    unique_combinations = set()
    while len(unique_combinations) < total_combinations:
        # generate a new combination
        # Randomly select half of the nodes from each layer
        random_proteomics_nodes = local_random.sample(proteomics_nodes, num_nodes)
        random_transcriptomics_nodes = [node.replace('.p', '.t') for node in random_proteomics_nodes]  # Get corresponding nodes in the other layer
        
        # Combine both sets of nodes and convert them to a tuple (immutable)
        random_nodes_tuple = tuple(random_proteomics_nodes + random_transcriptomics_nodes)
        unique_combinations.add(random_nodes_tuple)

    return list(unique_combinations)


def run_knockout_analysis(G_aggro, 
                          G_stable, 
                          knockout_type, 
                          knockout_target, 
                          red_range, 
                          t_values, 
                          orig_aggro_kernel, 
                          orig_nonmes_kernel, 
                          orig_gdd_values,  
                          pathway_df=None,
                          run_idx=None):
    """
    Performs knockouts on graphs, calculates diffusion distance and returns metrics.
    params:
        G_aggro: graph including aggressive subtype (cmsALL)
        G_stable: graph without CMS4 (cms123)
        knockout_type: knockout nodes, pathways or random collection of nodes
        red_range: range of expression reduction (knockdown vs knockout)
        t_values: time steps to run
        orig_aggro_kernel: Laplacian expontential kernel of aggressive subtype before knockout
        orig_nonmes_kernel: Laplacian expontential kernel of non-mes subtype before knockout
        orig_gdd_values: GDD values of original kernels
        pathway_df: dataframe with pathways and genes
        run_idx: index of the run for random pathway knockout
    returns:
        results: dictionary containing GDD values for the knockdowns
    """
    results = {}

    # knocking out specific nodes and pathways
    if knockout_type == 'runtype_node' or knockout_type == 'runtype_pathway':

        if knockout_type == 'runtype_pathway':
            target_list = pathway_df[pathway_df['Pathway'] == knockout_target]['genes'].str.split('|').explode().tolist()
            num_genes = len(target_list)  # Number of genes in the pathway
            print(f"Gene count in {knockout_target}: {num_genes}\n")


        results[knockout_target] = {}
        for reduction in red_range:
            # print(f"Processing {knockout_target} Knockdown with reduction factor: {reduction}")
            # Perform the knockout
            if args.symmetric:
                # Symmetric knockdown means that cms123 and cmsALL are knocked down equally
                knockdown_func = knockdown_node_both_layers if knockout_type == 'runtype_node' else knockdown_pathway_nodes

                # Get the knocked down (knockout) network and Laplacian for both cmsALL (aggressive) and cms123 (stable)
                knockdown_graph_aggro, knockdown_laplacian_aggro, _, _, _ = knockdown_func(G_aggro, knockout_target, reduced_weight=reduction)
                knockdown_graph_stable, knockdown_laplacian_stable, _, _, _ = knockdown_func(G_stable, knockout_target, reduced_weight=reduction)

                # Calculate diffusion kernels and GDD
                knock_aggro_eigs = laplacian_eigendecomp(knockdown_laplacian_aggro)
                knock_stable_eigs = laplacian_eigendecomp(knockdown_laplacian_stable)
                diff_kernel_knock_aggro = [knock_aggro_eigs[1] @ np.diag(np.exp(-t * knock_aggro_eigs[0])) @ knock_aggro_eigs[1].T for t in t_values]
                diff_kernel_knock_stable = [knock_stable_eigs[1] @ np.diag(np.exp(-t * knock_stable_eigs[0])) @ knock_stable_eigs[1].T for t in t_values]

                # Get GDD values
                gdd_values_trans = np.linalg.norm(np.array(diff_kernel_knock_stable) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')**2
                gdd_values_disruptA = np.linalg.norm(np.array(orig_aggro_kernel) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')**2
                gdd_values_disruptS = np.linalg.norm(np.array(orig_nonmes_kernel) - np.array(diff_kernel_knock_stable), axis=(1, 2), ord='fro')**2
            else:
                # Asymmetric knockdown, only cmsALL (aggressive) is knocked down
                knockdown_func = knockdown_node_both_layers if knockout_type == 'runtype_node' else knockdown_pathway_nodes

                knockdown_graph_aggro, knockdown_laplacian_aggro, _, _, _ = knockdown_func(G_aggro, knockout_target, reduced_weight=reduction)

                # Calculate diffusion kernels and GDD
                diff_kernel_knock_aggro = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian_aggro, t) for t in t_values]

                # Get GDD values
                gdd_values_trans = np.linalg.norm(np.array(orig_nonmes_kernel) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')**2
                gdd_values_disruptA = np.linalg.norm(np.array(orig_aggro_kernel) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')**2
                gdd_values_disruptS = np.linalg.norm(np.array(orig_nonmes_kernel) - np.array(diff_kernel_knock_stable), axis=(1, 2), ord='fro')**2


            results[knockout_target][reduction] = {
                'gdd_values_trans': gdd_values_trans,
                'gdd_values_disruptA': gdd_values_disruptA,
                'gdd_values_disruptS': gdd_values_disruptS,
                'max_gdd_trans': np.max(np.sqrt(gdd_values_trans)), # # get peak eta for symmetric, which corresponds to GDD
                'max_gdd_disruptS': np.max(np.sqrt(gdd_values_disruptA)), # get peak eta for stable disruption, which corresponds to GDD
                'max_gdd_disruptA': np.max(np.sqrt(gdd_values_disruptS)) # get peak eta for aggressive disruption, which corresponds to GDD
            }

            if args and args.visualize:
                results[knockout_target][reduction]['vis_kernels'] = [diff_kernel_knock_aggro[i] for i, t in enumerate(t_values)]

    # knocking out random nodes (ie random pathways)
    elif knockout_type == 'runtype_random':

        num_rand_nodes = knockout_target  # Number of genes in the pathway

        unique_node_combinations = generate_node_combinations(G_aggro, num_rand_nodes, total_combinations=args.permu_runs)

        for run_index in run_idx:

            node_list = unique_node_combinations[run_index]
            results[f'random_{num_rand_nodes}_run_{run_index}'] = {}

            for reduction in red_range:
                if args.symmetric: 
                    # Perform the knockout
                    knockdown_graph_aggro, knockdown_laplacian_aggro,_,_,_ = knockdown_random_nodes(G_aggro, node_list, reduced_weight=reduction)
                    knockdown_graph_stable, knockdown_laplacian_stable,_,_,_ = knockdown_random_nodes(G_stable, node_list, reduced_weight=reduction)

                    # Calculate diffusion kernels and GDD
                    knock_aggro_eigs = laplacian_eigendecomp(knockdown_laplacian_aggro)
                    knock_stable_eigs = laplacian_eigendecomp(knockdown_laplacian_stable)
                    diff_kernel_knock_aggro = [knock_aggro_eigs[1] @ np.diag(np.exp(-t * knock_aggro_eigs[0])) @ knock_aggro_eigs[1].T for t in t_values]
                    diff_kernel_knock_stable = [knock_stable_eigs[1] @ np.diag(np.exp(-t * knock_stable_eigs[0])) @ knock_stable_eigs[1].T for t in t_values]

                    gdd_values_trans = np.linalg.norm(np.array(diff_kernel_knock_stable) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')**2
                    gdd_values_disruptA = np.linalg.norm(np.array(orig_aggro_kernel) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')**2
                    gdd_values_disruptS = np.linalg.norm(np.array(orig_nonmes_kernel) - np.array(diff_kernel_knock_stable), axis=(1, 2), ord='fro')**2
                else:
                    # Perform the knockout
                    knockdown_graph_aggro, knockdown_laplacian_aggro,_,_,_ = knockdown_random_nodes(G_aggro, node_list, reduced_weight=reduction)

                    # Calculate diffusion kernels and GDD
                    diff_kernel_knock_aggro = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian_aggro, t) for t in t_values]

                    gdd_values_trans = np.linalg.norm(np.array(orig_nonmes_kernel) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')**2
                    gdd_values_disruptA = np.linalg.norm(np.array(orig_aggro_kernel) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')**2
                    gdd_values_disruptS = np.linalg.norm(np.array(orig_nonmes_kernel) - np.array(diff_kernel_knock_stable), axis=(1, 2), ord='fro')**2

                try:
                    results[f'random_{num_rand_nodes}_run_{run_index}'][reduction] = {
                        'max_gdd_trans': np.max(np.sqrt(gdd_values_trans)),
                        'max_gdd_disruptA': np.max(np.sqrt(gdd_values_disruptA)),
                        'max_gdd_disruptS': np.max(np.sqrt(gdd_values_disruptS))
                    }
                except KeyError as e:
                    print(f"KeyError encountered! Attempted to access results[{f'random_{num_rand_nodes}_run_{run_index}'}][{reduction}]")
                    print(f"The current keys in results are: {list(results.keys())}")
                    print(f"Contents of the problematic key if it exists: {results.get(f'random_{num_rand_nodes}_run_{run_index}', 'Key does not exist!')}")
                    print(f"Value of num_rand_nodes: {num_rand_nodes}, Value of _: {run_index}, Value of reduction: {reduction}")
                    raise e  # Re-raise the exception to halt the script and indicate error


    return results


# %%
################################################# NODE AND DIFFUSION PARAMETERS  #########################################
# get hubs and low nodes
degree_dict = dict(weighted_G_cms_ALL.degree(weighted_G_cms_ALL.nodes()))
# get nodes with largest degree and smallest degree
hub_nodes = sorted(degree_dict, key=lambda x: degree_dict[x], reverse=True)[:args.koh]
low_nodes = sorted(degree_dict, key=lambda x: degree_dict[x])[:args.kob]

if args.visualize:
    t_values = np.linspace(0.00, 10, 500)
else:
    t_values = np.linspace(0.01, 10, 250)

if args.koh == 0:
    nodes_to_investigate_bases = list(set([node.split('.')[0] for node in weighted_G_cms_ALL.nodes()])) # KNOCK OUT ALL NODES FOR FIXED REDUCTION, NODE COMPARISON
else:
    nodes_to_investigate_bases = list(set([node.split('.')[0] for node in hub_nodes + low_nodes])) # FOR FIXED REDUCTION, NODE COMPARISON


# DISTRIBUTE NODES ACROSS RANKS
if "SLURM_JOB_ID" in os.environ:
    # Distribute nodes across ranks
    nodes_subset = distribute_nodes(nodes_to_investigate_bases, rank, size)
    print(f'nodes for rank {rank}: {nodes_subset}')
else:
    nodes_subset = nodes_to_investigate_bases
    print(f'nodes for rank {rank}: {nodes_subset}')
    rank = 0
    size = 1

# if "SLURM_JOB_ID" not in os.environ:
    # args.test_net = True

# TEST NET
if args.test_net:
    # Create two multiplex graphs FOR TESTING
    weighted_G_cms_123 = create_multiplex_test(12)
    weighted_G_cms_ALL = create_multiplex_test(12)

    # Example nodes subset
    nodes_subset_with_suffix = list(weighted_G_cms_123.nodes())
    nodes_subset = list(set([node.split('.')[0] for node in nodes_subset_with_suffix]))
    print(f'nodes subset: {nodes_subset}')

    # Initialize containers for results
    orig_aggro_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_ALL), t) for t in t_values]
    orig_non_mesench_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_123), t) for t in t_values]
    orig_gdd_values = np.linalg.norm(np.array(orig_non_mesench_kernel) - np.array(orig_aggro_kernel), axis=(1, 2), ord='fro')**2
else:
    # Initialize containers for results
    orig_aggro_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_ALL), t) for t in t_values]
    orig_non_mesench_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_123), t) for t in t_values]
    orig_gdd_values = np.linalg.norm(np.array(orig_non_mesench_kernel) - np.array(orig_aggro_kernel), axis=(1, 2), ord='fro')**2


# get max orig_gdd_values
max_orig_gdd_values = np.max(np.sqrt(orig_gdd_values))

# print(f'max_orig_gdd_values: {max_orig_gdd_values}')









# %% 
####################################### RUN # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
start_time = time.time()

local_target_results = {}

# if not "SLURM_JOB_ID" in os.environ:
#     args.pathway = False
#     args.koh = 0

if args.pathway == True:
    # PATHWAY KNOCKOUTS
    for pathway in pathways_subset:
        pathway_results = run_knockout_analysis(weighted_G_cms_ALL, weighted_G_cms_123, 'runtype_pathway', pathway, red_range, t_values, orig_aggro_kernel, orig_non_mesench_kernel, orig_gdd_values, pathway_df)
        local_target_results.update(pathway_results)

    if args.permu_runs:
        # RANDOM PATHWAY KNOCKOUTS (for permutation analysis)
        local_rand_results = {}
        
        # Determine the subset of run indices for each rank
        run_indices = distribute_runs(args.permu_runs, rank, size)

        pathway_sizes = args.path_size_range.split(',')
        pathway_sizes = range(int(pathway_sizes[0]), int(pathway_sizes[1]) + 1)
        
        # print(f'pathway sizes for rank {rank}: {pathway_sizes}')
        print(f'run indices for rank {rank}: {run_indices}')

        # Create a boolean series where each element is True if the 'Pathway' column contains any of the pathway descriptions
        for random_pathway_size in pathway_sizes:
            rand_results = run_knockout_analysis(weighted_G_cms_ALL, weighted_G_cms_123, 'runtype_random', random_pathway_size, red_range, t_values, orig_aggro_kernel, orig_non_mesench_kernel, orig_gdd_values, pathway_df, run_idx=run_indices)
            
            local_rand_results.update(rand_results)
            
        print(f'Rank {rank} has finished the target and random pathway runs.')


else:
    # NODE knockouts
    with tqdm(total=len(nodes_subset), desc=f"Calculating {len(nodes_subset)} gene knockouts, progress") as pbar:
        for node in nodes_subset:
            # RUN the knockout analysis function
            node_results = run_knockout_analysis(weighted_G_cms_ALL, weighted_G_cms_123, 'runtype_node', node, red_range, t_values, orig_aggro_kernel, orig_non_mesench_kernel, orig_gdd_values)
            local_target_results.update(node_results)
            pbar.update(1)


# GATHERING RESULTS
all_target_results = comm.gather(local_target_results, root=0)

# FOR PATHWAY KNOCKOUTS
if args.pathway:
    if args.permu_runs:
        all_rand_results = comm.gather(local_rand_results, root=0)
        all_results_list = [all_target_results, all_rand_results]
        filename_identifiers = ['target', 'random']
    else:
        all_results_list = [all_target_results]
        filename_identifiers = ['target']
    
    # Assuming 'pathways' is a list of strings
    unique_identifier = ''.join([pathway[0] for pathway in pathways])
    unique_identifier = unique_identifier[:20]

# FOR NODE KNOCKOUTS
else:
    all_results_list = [all_target_results]
    filename_identifiers = ['target']

    unique_identifier = ''.join([node[0] for node in nodes_to_investigate_bases])
    unique_identifier = unique_identifier[:5]

for i, all_results in enumerate(all_results_list):
    # Post-processing on the root processor
    if rank == 0 and "SLURM_JOB_ID" in os.environ:
        # Initialize a master dictionary to combine results
        combined_results = {}

        # Combine the results from each process
        for process_results in all_results:
            for key, value in process_results.items():
                combined_results[key] = value

        with open(f'diff_results/Pathway_{args.pathway}_{filename_identifiers[i]}_{unique_identifier}_GDDs_ks{str(orig_aggro_kernel[0].shape[0])}_permu{args.permu_runs}_symmetric{args.symmetric}_{args.net_dens}_{args.path_size_range}.pkl', 'wb') as f:
            pkl.dump(combined_results, f)
        
        os.system("cp -r diff_results/ $HOME/thesis_code/Diffusion/")
        print('Saving has finished.')


    elif rank == 0 and "SLURM_JOB_ID" not in os.environ:
        with open(f'diff_results/LOCAL_Pathway_{args.pathway}_{filename_identifiers[i]}_{unique_identifier}_GDDs_ks{str(orig_aggro_kernel[0].shape[0])}_permu{args.permu_runs}_symmetric{args.symmetric}_{args.net_dens}_{args.path_size_range}.pkl', 'wb') as f:
            pkl.dump(local_target_results, f)

        # with open(f'diff_results/Pathway_{args.pathway}_random_node_{unique_identifier}_GDDs_ks{str(orig_aggro_kernel[0].shape[0])}.pkl', 'wb') as f:
        #     pkl.dump(local_rand_results, f)


# get the end time
end_time = time.time()
print(f'elapsed time (node knockdown calc) (rank {rank}): {int(end_time - start_time)}s')



MPI.Finalize()







# %% ########################################### Analysis and Visualisation ##########################################

if "SLURM_JOB_ID" not in os.environ:

    #  NODE KNOCKOUT ANALYSIS
    t_values = np.linspace(0.01, 10, 250)
    if args.pathway == False:
        with open(f'diff_results/LOCAL_Pathway_False_target_XPSPY_GDDs_ks308_permuNone_symmetricTrue_low_dens_5,26.pkl', 'rb') as f:
            node_knockouts = pkl.load(f)

        print(f'node_knockouts: {node_knockouts.keys()}')
        print(f'Reduction factors: {node_knockouts[list(node_knockouts.keys())[0]].keys()}')
        # print(f'GDD values: {node_knockouts[list(node_knockouts.keys())[0]][0.05]}')


        # for key in node_knockouts.keys():
        #     print(f'node {key}: {node_knockouts[key].keys()}')
        # Choose the node and t_values for plotting
        selected_node = np.random.choice(list(node_knockouts.keys()))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=300)

        # Left plot: GDD values over time for various reductions (single node)
        for reduction in node_knockouts[selected_node].keys():
            gdd_values_trans = node_knockouts[selected_node][reduction]['gdd_values_trans']
            gdd_values_disruptA = node_knockouts[selected_node][reduction]['gdd_values_disruptA']
            gdd_values_disruptS = node_knockouts[selected_node][reduction]['gdd_values_disruptS']
            ax1.plot(t_values, gdd_values_trans, label=f'Reduction {reduction}')
            ax1.plot(t_values, gdd_values_disruptA, label=f'Reduction {reduction} (Disrupt)')

        ax1.set_title(f'GDD Over Time for Various Reductions\nNode: {selected_node}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('GDD Value')
        ax1.legend()
        ax1.grid(True)

        # Choose a reduction factor from the list of reductions
        selected_reduction = red_range[0]
        selected_reduction = 0.05

        max_gdds_trans = {}
        max_gdds_disruptA = {}
        max_gdds_disruptS = {}
        # Right plot: GDD values over time for a single reduction (all nodes)
        for node_base in node_knockouts.keys():
            gdd_values_trans = node_knockouts[node_base][selected_reduction]['gdd_values_trans']
            gdd_values_disruptA = node_knockouts[node_base][selected_reduction]['gdd_values_disruptA']
            gdd_values_disruptS = node_knockouts[node_base][selected_reduction]['gdd_values_disruptS']
            ax2.plot(t_values, gdd_values_disruptA, label=f'Node {node_base}', alpha=0.5)
            # ax2.plot(t_values, gdd_values_disruptA, label=f'Node {node_base} (Disrupt)', alpha=0.5)
            max_gdds_trans[node_base] = np.max(gdd_values_trans)
            max_gdds_disruptA[node_base] = np.max(gdd_values_disruptA)
            max_gdds_disruptS[node_base] = np.max(gdd_values_disruptS)

        ax2.set_title(f'GDD Over Time for Single Reduction\nReduction: {selected_reduction}')
        ax2.set_xlabel('Time')
        # ax2.set_ylabel('GDD Value')  # Y-label is shared with the left plot
        # ax2.legend()
        ax2.set_xlim([0, 2])
        ax2.grid(True)

        plt.show()

        # %%
        # print max_gdds in ascending order
        sorted_max_gdds_trans = {k: v for k, v in sorted(max_gdds_trans.items(), key=lambda item: item[1])}
        sorted_max_gdds_disruptA = {k: v for k, v in sorted(max_gdds_disruptA.items(), key=lambda item: item[1], reverse=True)}
        sorted_max_gdds_disruptS = {k: v for k, v in sorted(max_gdds_disruptS.items(), key=lambda item: item[1], reverse=True)}

        # add key for original max_gdd (Which is the original distance between the aggresive subtype and stable subtype)
        sorted_max_gdds_disruptA['ORIGINAL_MAX_GDD'] = max_orig_gdd_values

        # print original max_gdd 
        # print(f'Original GDD between aggressive and nonmesenchymal: {max_orig_gdd_values}')
        # print(f'GDD values of top transition node knockouts (maximum distance decrease): {sorted_max_gdds_trans}')

        num_top_genes = 10
        # # Print first N items of sorted_max_gdds_disruptA
        print(f'GDD values of top disruptor node knockouts (maximum self-distance increase):')
        for key, value in list(sorted_max_gdds_disruptA.items())[:num_top_genes]:
            print(f'{key}: {value}')

        print('-------------------')

        # for key, value in list(sorted_max_gdds_disruptS.items())[:num_top_genes]:
        #     print(f'{key}: {value}')

        # print the length of the overlap of the top 10 nodes
        overlap = set(list(sorted_max_gdds_disruptA.keys())[:num_top_genes]).intersection(set(list(sorted_max_gdds_disruptS.keys())[:num_top_genes]))

        # print(f'overlap: {len(overlap)}')

        # print the ones that are not in the overlap
        print('The following knockouts significantly disrupt the aggressive subtype, but not the stable subtype: ')
        for key, value in list(sorted_max_gdds_disruptA.items())[:num_top_genes]:
            if key not in overlap:
                print(f'{key}: {value}')



        # %%
        summed_degrees = {}
        summed_degrees_aggro = {}
        summed_betweenness = {}
        summed_betweenness_aggro = {}
        for node in sorted_max_gdds_trans.keys():
            node_p = node + '.p'
            node_t = node + '.t'
            
            # Accessing the degree for each node directly from the DegreeView
            degree_p_aggro = weighted_G_cms_ALL.degree(node_p, weight='weight') if node_p in weighted_G_cms_ALL else 0
            degree_t_aggro = weighted_G_cms_ALL.degree(node_t, weight='weight') if node_t in weighted_G_cms_ALL else 0
            degree_p_stable = weighted_G_cms_123.degree(node_p, weight='weight') if node_p in weighted_G_cms_123 else 0
            degree_t_stable = weighted_G_cms_123.degree(node_t, weight='weight') if node_t in weighted_G_cms_123 else 0

            summed_betweenness = nx.betweenness_centrality(weighted_G_cms_ALL, weight='weight')
            summed_betweenness_aggro[node] = summed_betweenness[node_p] + summed_betweenness[node_t]
            
            summed_degree_aggro = degree_p_aggro + degree_t_aggro
            summed_degree_stable = degree_p_stable + degree_t_stable
            summed_degrees[node] = summed_degree_aggro + summed_degree_stable / 4
            summed_degrees_aggro[node] = summed_degree_aggro / 4


        # Make dataframe with node, max_gdd, and summed_degree
        df = pd.DataFrame.from_dict(sorted_max_gdds_trans, orient='index', columns=['max_gdd_trans'])
        # add row with original max_gdd
        df.loc['ORIGINAL_MAX_GDD'] = max_orig_gdd_values

        df['max_gdd_disrupt'] = sorted_max_gdds_disruptA.values()
        # make column with max_gdd delta
        df['max_gdd_delta_trans'] = df['max_gdd_trans'] - df.loc['ORIGINAL_MAX_GDD', 'max_gdd_trans']
        df['max_gdd_delta_disrupt'] = df['max_gdd_disrupt'] - df.loc['ORIGINAL_MAX_GDD', 'max_gdd_disrupt']
        # make column with max_gdd / original_max_gdd
        # df['max_gdd_ratio_trans'] = df['max_gdd_trans'] / df.loc['ORIGINAL_MAX_GDD', 'max_gdd_trans']
        # df['max_gdd_ratio_disrupt'] = df['max_gdd_disrupt'] / df.loc['ORIGINAL_MAX_GDD', 'max_gdd_disrupt']
        df['summed_degree'] = df.index.map(summed_degrees)
        df['summed_degree_aggro'] = df.index.map(summed_degrees_aggro)
        df['summed_betweenness_aggro'] = df.index.map(summed_betweenness_aggro)
        df = df.sort_values(by='max_gdd_trans', ascending=True)

        print(df.head())

        # plot summed_degrees_aggro vs max_gdd_disrupt
        plt.figure(figsize=(10, 6))
        plt.scatter(df['summed_degree_aggro'], df['max_gdd_disrupt'])
        plt.xlabel('Summed Degree (Aggressive)')
        plt.ylabel('Max GDD (Disrupt)')
        plt.title('Summed Degree (Aggressive) vs Max GDD (Disrupt)')
        # plt.ylim([1.95, 2.05])
        plt.show()

        print(df['summed_degree_aggro'])

        # # plot summed_betweenness_aggro vs max_gdd_disrupt
        # plt.figure(figsize=(10, 6))
        # plt.scatter(df['summed_betweenness_aggro'], df['max_gdd_disrupt'])
        # plt.xlabel('Summed Betweenness (Aggressive)')
        # plt.ylabel('Max GDD (Disrupt)')
        # plt.title('Summed Betweenness (Aggressive) vs Max GDD (Disrupt)')
        # plt.ylim([1.95, 2.05])
        # plt.show()

        # # plot the distribution of max_gdd_delta_trans
        # plt.figure(figsize=(10, 6))
        # plt.hist()



        # write to file
        df.to_csv(f'diff_results/NODE_KNOCKOUTS_RESULTS_symmetric{args.symmetric}_{args.net_dens}.csv', index=True)

        # %% SAVE FOR LATER
        # PLOT 2 (WEAKER KNOCKDOWN)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=300)


        # make another plot as ax2 but for reduction factor = 0.8
        selected_reduction = red_range[-1]

        max_gdds = {}
        # Right plot: GDD values over time for a single reduction (all nodes)
        for node_base in node_knockouts.keys():
            gdd_values = node_knockouts[node_base][selected_reduction]['gdd_values']
            ax2.plot(t_values, gdd_values, label=f'Node {node_base}', alpha=0.5)
            max_gdds[node_base] = np.max(gdd_values)


        ax2.set_title(f'GDD Over Time for Single Reduction\nReduction: {selected_reduction}')
        ax2.set_xlabel('Time')
        # ax2.set_ylabel('GDD Value')  # Y-label is shared with the left plot
        # ax2.legend()
        ax2.set_xlim([0, 2])
        ax2.grid(True)

        plt.show()
        # print(max_gdds)
        # max_GDD_1 = max_gdds['1']
        # max_GDD_2 = max_gdds['2']
        # print(max_GDD_1 - max_GDD_2)

        selected_reduction = red_range[-1]
        # order nodes by max GDD
        max_gdds = {}
        for node_base in node_knockouts.keys():
            max_gdds[node_base] = np.max(node_knockouts[node_base][selected_reduction]['gdd_values'])

        sorted_max_gdds = {k: v for k, v in sorted(max_gdds.items(), key=lambda item: item[1])}

        # get the nodes with the highest GDD
        highest_gdd_nodes = list(sorted_max_gdds.keys())[-5:]
        highest_gdd_nodes



















    # %%
    if "SLURM_JOB_ID" not in os.environ:
        # VISUALIZE DIFFUSION
        args.visualize = True

        def multiplex_net_viz(M, ax, diff_colors=False, node_colors=None, node_sizes=None, node_coords=None):

            dark_red = "#8B0000"  # Dark red color
            dark_blue = "#00008B"  # Dark blue color

            # Iterate over nodes in the 'PROTEIN' layer
            # Initialize a dictionary to hold node colors
            for node in M.iter_nodes(layer='PROTEIN'):
                if not diff_colors:
                    node_colors[(node, 'PROTEIN')] = dark_red

            # Iterate over nodes in the 'RNA' layer
            for node in M.iter_nodes(layer='RNA'):
                if not diff_colors:
                    node_colors[(node, 'RNA')] = dark_blue


            edge_color = "#505050"  # A shade of gray, for example

            # Initialize a dictionary to hold edge colors
            edge_colors = {}

            # Assign colors to edges in the 'PROTEIN' and 'RNA' layers
            # Assuming edges are between nodes within the same layer
                
            layer_colors = {'PROTEIN': "red", 'RNA': "blue"}


            fig = pn.draw(net=M,
                    ax=ax,
                    show=False, 
                    nodeColorDict=node_colors,
                    nodeLabelDict={nl: None for nl in M.iter_node_layers()},  # Set all node labels to None explicitly
                    nodeLabelRule={},  # Clear any label rules
                    defaultNodeLabel=None,  # Set default label to None
                    nodeSizeDict=node_sizes,
                    # nodeSizeRule={"rule":"degree", "scalecoeff":0.00001},
                    nodeCoords=node_coords,
                    edgeColorDict=edge_colors,
                    defaultEdgeAlpha=0.08,
                    layerColorDict=layer_colors,
                    defaultLayerAlpha=0.075,
                    # camera_dist=6,
                    # layerLabelRule={},  # Clear any label rules
                    # defaultLayerLabel=None,  # Set default label to None
                    # azim=45,
                    # elev=25
                    )

            # print(type(fig))

            return fig

        # %%

        t_values = np.linspace(0, 10, 500)
        # visualisation_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_ALL), t) for t in t_values_viz]


        def multiplex_diff_viz(M, weighted_G, ax=None, node_colors=None, node_sizes=None):
            # Load the pickle file
            with open('diff_results/Pathway_False_target_X_GDDs_ks308_permuNone_symmetricTrue_low_dens.pkl', 'rb') as f: # 'diff_results/Pathway_False_target_BB_GDDs_ks272.pkl'
                results = pkl.load(f)
            
            time_resolved_kernels = results['XRCC1'][0.05]['vis_kernels'] #  visualisation_kernel[:7] # results['ACACA'][0.00]['vis_kernels'][:12] # BIRC2

            max_gdd = 'max_gdd_disrupt'

            max_gdd = results['XRCC1'][0.05][max_gdd] # 0 # results['ACACA'][0.00]['max_gdd_index'] # BIRC2
            gdd_values_trans = results['XRCC1'][0.05]['gdd_values_trans']
            max_gdd_index = np.argmax(np.sqrt(gdd_values_trans) == max_gdd)
            max_gdd_time = t_values[max_gdd_index]
            # Get the corresponding kernel from vis_kernels
            corresponding_kernel = results['XRCC1'][0.05]['vis_kernels'][max_gdd_index]
            time_points_to_plot = [0, max_gdd_time]  # Initial and max GDD time
            # Add a time point after max_gdd_time, for example, one step further in the t_values array
            # Calculate the next index
            next_index = min(len(t_values)-1, 75)
            # Add the time point to the list
            time_points_to_plot.append(t_values[next_index])
            # Extract the kernels for the selected time points
            selected_kernels = [results['XRCC1'][0.05]['vis_kernels'][int(np.round(time_point * (len(t_values) - 1) / 10))] for time_point in time_points_to_plot]


            # Assume 'weighted_G_cms_ALL' is your graph
            node_order = list(weighted_G.nodes())
            orphans = ['ACACA.p', 'MRE11A.p', 'NFKB1.p', 'CTNNA1.p']
            nodes_with_degree = [node for node in node_order if node not in orphans]
            # Create a subgraph with these nodes
            subgraph = weighted_G_cms_ALL.subgraph(nodes_with_degree)
            # Now calculate the layout using only the nodes in the subgraph
            pos = nx.spring_layout(subgraph)
            # The orphan nodes will be assigned a default position as they are not included in the subgraph
            prot_node_positions = pos.copy()
            for node in weighted_G_cms_ALL.nodes():
                if node not in prot_node_positions and node.endswith('.p'):
                    prot_node_positions[node] = (0,0)  # or any other default position

            # Define the suffixes for each layer
            suffixes = {'PROTEIN': '.p', 'RNA': '.t'}

            # Create an index mapping from the suffixed node name to its index
            node_indices = {node: i for i, node in enumerate(node_order)}
            node_colors = {}  
            node_sizes = {} 
            node_coords = {}

            for node, pos in prot_node_positions.items():
                stripped_node = node.rstrip('.p')  # Remove the suffix to match identifiers in M
                node_coords[stripped_node] = tuple(pos)

            max_deg = 28
            min_deg = 1
            max_scaled_size = 3.5
            for nl in M.iter_node_layers():  # Iterating over all node-layer combinations
                node, layer = nl  # Split the node-layer tuple
                neighbors = list(M._iter_neighbors_out(nl, dims=None))  # Get all neighbors for the node-layer tuple
                degree = len(neighbors)  # The degree is the number of neighbors
                normalized_degree = (degree - min_deg) / (max_deg - min_deg)
                scaled_degree = 1 + normalized_degree * (max_scaled_size - 1)

                # Assign to node sizes with scaling factor
                node_sizes[nl] = scaled_degree * 0.02


            # Set up a 3x3 subplot grid with 3D projection
            fig = plt.figure(figsize=(15, 8))

            axs_diffusion = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(3)]
            axs_gdd = [fig.add_subplot(2, 3, i + 4) for i in range(3)]

            j = 0
            global_max = max(kernel.max() for kernel in time_resolved_kernels)
            norm = Normalize(vmin=0, vmax=1)
            # print(global_max)
            # Ensure you have a list of the 25 time-resolved kernels named 'time_resolved_kernels'
            for ax, kernel, time_point in zip(axs_diffusion, selected_kernels, time_points_to_plot):
                # Create the unit vector e_j with 1 at the jth index and 0 elsewhere
                e_j = np.zeros(len(weighted_G.nodes()))
                e_j[j] = 100

                
                # Multiply the kernel with e_j to simulate diffusion from node j
                diffusion_state = kernel @ e_j
                # order the diffusion state in descending order
                diffusion_state = sorted(diffusion_state, reverse=True)

                # Now, update node colors and sizes based on diffusion_state
                for layer in M.iter_layers():  # Iterates through all nodes in M
                    # Determine the layer of the node for color and size settings
                    for node in M.iter_nodes(layer=layer):
                        # Append the appropriate suffix to the node name to match the format in node_order
                        suffixed_node = node + suffixes[layer]
                        # Use the suffixed node name to get the corresponding index from the node_order
                        index = node_indices[suffixed_node]

                        # Map the diffusion state to a color and update node_colors and node_sizes
                        color = plt.cm.viridis(diffusion_state[index])  # Mapping color based on diffusion state
                        node_colors[(node, layer)] = color
                        # node_sizes[(node, layer)] = 0.03  # Or some logic to vary size with diffusion state

                # Now use your updated visualization function with the new colors and sizes
                diff_fig = multiplex_net_viz(M, ax, diff_colors=True, node_colors=node_colors, node_sizes=node_sizes, node_coords=node_coords)

                ax.set_title(f"T = {time_point:.2f}")

            for ax, time_point in zip(axs_gdd, time_points_to_plot):
                # Plot all gdd_values_trans
                ax.plot(t_values[:150], gdd_values_trans[:150])

                # Add a vertical line at the corresponding time_point
                ax.axvline(x=time_point, color='red', linestyle='dotted', label=f'Time Point', linewidth=2)

                # ax.set_title(f"GDD values with time point T = {time_point:.2f}")
                # ax.set_xlabel('Time')
                # ax.set_ylabel('GDD Value')
                if time_point == 0:
                    ax.legend()
                    ax.set_ylabel('GDD Value')
                elif time_point == max_gdd_time:
                    ax.set_xlabel('Time')
                # ax.legend()

            # Adjust layout to prevent overlap
            # plt.tight_layout()
            # plt.subplots_adjust(top=0.95)  # Adjust the top space to fit the main title
            # plt.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
            # plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), cax=cbar_ax, orientation='vertical', label='Concentration')
            # savethefigure
            plt.savefig('diffusion_figure.svg') # make rc.param no font export

            # Display the figure
            plt.tight_layout()
            plt.show()

        if args.visualize:
            multiplex_diff_viz(pymnet_ALL, weighted_G_cms_ALL)


    # %%

    # 4x4 diffusion plot
    t_values_viz = np.linspace(0, 3, 10)
    visualisation_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_ALL), t) for t in t_values_viz]


    def multiplex_diff_viz(M, weighted_G, ax=None, node_colors=None, node_sizes=None):
        # Load the pickle file
        with open('diff_results/Pathway_False_target_XAMNC_GDDs_ks308_permuNone_symmetricTrue_low_dens.pkl', 'rb') as f: # 'diff_results/Pathway_False_target_BB_GDDs_ks272.pkl'
            results = pkl.load(f)
        
        # with open('diff_results/Pathway_False_target_BCGGS_GDDs_ks308_permuNone.pkl', 'rb') as f:
        #     results = pkl.load(f)
        
        time_resolved_kernels =  visualisation_kernel[:5] # results['ACACA'][0.00]['vis_kernels'][:12] # BIRC2

        # Assume 'weighted_G_cms_ALL' is your graph
        node_order = list(weighted_G.nodes())
        # get indices of nodes that end with '.t'
        t_indices = [i for i, node in enumerate(node_order) if node.endswith('.t')]
        #consistent node positions

        orphans = ['ACACA.p', 'MRE11A.p', 'NFKB1.p', 'CTNNA1.p']
        nodes_with_degree = [node for node in node_order if node not in orphans]

        # Create a subgraph with these nodes
        subgraph = weighted_G_cms_ALL.subgraph(nodes_with_degree)

        # Now calculate the layout using only the nodes in the subgraph
        pos = nx.spring_layout(subgraph)

        # print(pos)

        # If you want to use the same positions for the nodes in the original graph, you can do so. 
        # The orphan nodes will be assigned a default position as they are not included in the subgraph.
        prot_node_positions = pos.copy()
        for node in weighted_G_cms_ALL.nodes():
            if node not in prot_node_positions and node.endswith('.p'):
                prot_node_positions[node] = (0,0)  # or any other default position


        # Set up a 3x3 subplot grid with 3D projection
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10), subplot_kw={'projection': '3d'})
        axs = axes.flatten()  # Flatten the axes array for easy iteration
        # fig.suptitle('Network Diffusion over 25 Time Points')
        

        # Define the suffixes for each layer
        suffixes = {'PROTEIN': '.p', 'RNA': '.t'}

        # Create an index mapping from the suffixed node name to its index
        node_indices = {node: i for i, node in enumerate(node_order)}
        node_colors = {}  
        node_sizes = {} 
        node_coords = {}

        for node, pos in prot_node_positions.items():
            stripped_node = node.rstrip('.p')  # Remove the suffix to match identifiers in M
            node_coords[stripped_node] = tuple(pos)
            # node_coords[stripped_node] = (1,1)


        max_deg = 28
        min_deg = 1
        max_scaled_size = 3.5
        for nl in M.iter_node_layers():  # Iterating over all node-layer combinations
            node, layer = nl  # Split the node-layer tuple
            neighbors = list(M._iter_neighbors_out(nl, dims=None))  # Get all neighbors for the node-layer tuple
            degree = len(neighbors)  # The degree is the number of neighbors
            normalized_degree = (degree - min_deg) / (max_deg - min_deg)
            scaled_degree = 1 + normalized_degree * (max_scaled_size - 1)

            # Assign to node sizes with your scaling factor
            node_sizes[nl] = scaled_degree * 0.02  # Adjust the scaling factor as needed



        j = 0
        global_max = max(kernel.max() for kernel in time_resolved_kernels)
        norm = Normalize(vmin=0, vmax=1)
        # print(global_max)
        # Ensure you have a list of the 25 time-resolved kernels named 'time_resolved_kernels'
        for idx, (ax, kernel) in enumerate(zip(axs, time_resolved_kernels[:4])):
            # if idx % 5 == 0:
            # Create the unit vector e_j with 1 at the jth index and 0 elsewhere
            e_j = np.zeros(len(weighted_G.nodes()))
            e_j[j] = 100

            # e_j = np.ones(len(weighted_G_cms_ALL.nodes()))
            
            # Multiply the kernel with e_j to simulate diffusion from node j
            diffusion_state = kernel @ e_j
            # order the diffusion state in descending order
            diffusion_state = sorted(diffusion_state, reverse=True)

            # Now, update node colors and sizes based on diffusion_state
            for layer in M.iter_layers():  # Iterates through all nodes in M
                # Determine the layer of the node for color and size settings
                for node in M.iter_nodes(layer=layer):
                    # Append the appropriate suffix to the node name to match the format in node_order
                    suffixed_node = node + suffixes[layer]
                    # Use the suffixed node name to get the corresponding index from the node_order
                    index = node_indices[suffixed_node]

                    # Map the diffusion state to a color and update node_colors and node_sizes
                    color = plt.cm.viridis(diffusion_state[index])  # Mapping color based on diffusion state
                    node_colors[(node, layer)] = color
                    # node_sizes[(node, layer)] = 0.03  # Or some logic to vary size with diffusion state

            # Now use your updated visualization function with the new colors and sizes
            diff_fig = multiplex_net_viz(M, ax, diff_colors=True, node_colors=node_colors, node_sizes=node_sizes, node_coords=node_coords)

            ax.set_title(f"T = {idx * (6/25):.2f}")

        # Adjust layout to prevent overlap
        # plt.tight_layout()
        # plt.subplots_adjust(top=0.95)  # Adjust the top space to fit the main title
        plt.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), cax=cbar_ax, orientation='vertical', label='Concentration')
        # savethefigure
        # plt.savefig('diffusion_figure.png', dpi=600) # make rc.param no font export

        # Display the figure
        plt.tight_layout()
        # save as svg
        plt.savefig('diffusion_figure2.svg') # make rc.param no font export
        plt.show()

    args.visualize = True
    if args.visualize:
        multiplex_diff_viz(pymnet_ALL, weighted_G_cms_ALL)

    # %%
    # make a standalone figure of the colorbar
    fig, ax = plt.subplots(figsize=(1, 6), dpi = 300)
    norm = Normalize(vmin=0, vmax=1)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), cax=ax, orientation='vertical', label='Concentration')

    plt.show()
    plt.savefig('colorbar.svg')


    # %% COMPARING DIRECT KERNEL WITH KERNEL EIGENDECOMPOSITION
    # # start time
    # start_time = time.time()
    # # calculate laplacian_exponential_diffusion_kernel for knockdown graph for 5 different t values
    # knockdown_kernel = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian, t) for t in t_values]

    # mid_time = time.time()
    # eigentimes = mid_time - start_time

    # # check whether we get same result for laplacian_exponential_diffusion_kernel
    # knockdown_kernel_direct = [laplacian_exponential_diffusion_kernel(knockdown_laplacian, t) for t in t_values]

    # # end time
    # end_time = time.time()
    # directtimes = end_time - mid_time

    # np.allclose(knockdown_kernel, knockdown_kernel_direct)

    # print(f'Eigen-decomposition time: {eigentimes}')
    # print(f'Direct computation time: {directtimes}')

    # %%
    # from scipy.sparse.linalg import eigsh
    # from scipy.sparse import csr_matrix, issparse

    # def force_dense_eig_solver(L):
    #     if issparse(L):
    #         L = L.toarray()  # Convert sparse matrix to dense
    #     eigenvalues, eigenvectors = np.linalg.eigh(L)  # Use dense method
    #     return eigenvalues, eigenvectors

    # def force_sparse_eig_solver(L, k=6):  # You can adjust 'k' as needed
    #     # Ensure L is a sparse matrix for eigsh, if not, convert it.
    #     if not issparse(L):
    #         L = csr_matrix(L)  # Convert dense matrix to sparse format, choose appropriate sparse format like csr or csc
    #     eigenvalues, eigenvectors = eigsh(L, k=k)  # Use sparse method
    #     return eigenvalues, eigenvectors


    # def laplacian_exponential_kernel_eigendecomp(L, t, sparse=False):
    #     """
    #     Function to compute the Laplacian exponential diffusion kernel using eigen-decomposition
    #     The function takes the Laplacian matrix L and a time parameter t as inputs.
    #     """
    #     # Calculate the eigenvalues and eigenvectors of the Laplacian matrix
    #     if sparse:
    #         N = L.shape[0]
    #         eigenvalues, eigenvectors = force_sparse_eig_solver(L, k=N-1)
    #     else:
    #         eigenvalues, eigenvectors = force_dense_eig_solver(L)
    #     # Compute the matrix exponential using eigenvalues
    #     exp_eigenvalues = np.exp(-t * eigenvalues)
    #     # Reconstruct the matrix using the eigenvectors and the exponentiated eigenvalues
    #     kernel = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T

    #     return kernel


    # t_values = np.linspace(0.01, 10, 50)
    # reduction = 0.05

    # for knockout_target in pathways_subset:

    #     t = 0.5

    #     start_time = time.time()
    #     knockdown_graph_aggro, knockdown_laplacian, _, _, _ = knockdown_pathway_nodes(weighted_G_cms_ALL, knockout_target, reduced_weight=reduction)

    #     # Calculate laplacian exponential diffusion kernel for knockdown graph for different t values using eigen-decomposition
    #     # knockdown_kernel_eigendecomp = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian, t, sparse=False) for t in t_values]
    #     knockdown_kernel_eigendecomp = laplacian_exponential_kernel_eigendecomp(knockdown_laplacian, t, sparse=False)


    #     # Mid-time between the methods
    #     mid_time = time.time()

    #     # knockdown_sparse_kernel_eigendecomp = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian, t, sparse=True) for t in t_values]
    #     knockdown_sparse_kernel_eigendecomp = laplacian_exponential_kernel_eigendecomp(knockdown_laplacian, t, sparse=True)

    #     mid2_time = time.time()

    #     # Calculate laplacian exponential diffusion kernel for knockdown graph for different t values using direct method
    #     # knockdown_kernel_direct = [laplacian_exponential_diffusion_kernel(knockdown_laplacian, t) for t in t_values]
    #     knockdown_kernel_direct = laplacian_exponential_diffusion_kernel(knockdown_laplacian, t)
        

    #     # End time for direct method
    #     end_time = time.time()

    #     eigen_times = mid_time - start_time
    #     eigen_times_sparse = mid2_time - mid_time
    #     direct_times = end_time - mid2_time

    #     # Check if the results from both methods are close to each other
    #     are_close = np.allclose(knockdown_kernel_eigendecomp, knockdown_kernel_direct)

    #     are_close_sparse = np.allclose(knockdown_sparse_kernel_eigendecomp, knockdown_kernel_eigendecomp, atol=1e-1, rtol=1e-1)

    #     if are_close_sparse == False:
    #         # Print time taken by each method and the result of the comparison
    #         print(f'Eigen-decomposition time: {eigen_times}')
    #         print(f'Eigen-decomposition time (sparse): {eigen_times_sparse}')
    #         print(f'Direct computation time: {direct_times}\n')
    #         print(f'Are the results from direct and eigendecomp methods close? {are_close}')
    #         print(f'Are the results from sparse and eigendecomp methods close? {are_close_sparse}')

    #         A = knockdown_sparse_kernel_eigendecomp
    #         B = knockdown_kernel_eigendecomp

    #         C = knockdown_kernel_direct

    #         # Compute the Frobenius norm of the difference
    #         frobenius_norm = np.linalg.norm(A - B, 'fro')

    #         print("The Frobenius norm of the difference is:", frobenius_norm)

    #         frob_norm_direct = np.linalg.norm(B - C, 'fro')

    #         print("The Frobenius norm of the difference is (direct to eigenfull):", frob_norm_direct)




    # %%
    ### BELOW HERE IS THE CODE GRAVEYARD ##########################################



    # %% ############################ BARBELL GRAPHS ############################
    # The first graph will be a complete barbell graph, the second will have its bridge removed, 
    # and the third will have its central connection removed.

    t_values = np.linspace(0, 10, 500)

    # Define the number of nodes in the barbell graph's complete subgraphs and the bridge length
    m1 = 5  # Number of nodes in the complete subgraphs
    m2 = 0  # Number of nodes in the bridge

    # Generate the complete barbell graph
    G_single_edge = nx.barbell_graph(m1, m2)

    # Identify the nodes to move and disconnect
    node_to_move_from_first_bell = m1 - 2  # Second to last node in the first bell
    node_to_move_from_second_bell = m1 + 1  # Second node in the second bell

    G_complete = G_single_edge.copy()
    # Add the new edge directly connecting the two identified nodes
    # G_complete.add_edge(node_to_move_from_first_bell, node_to_move_from_second_bell)

    # Verify the graphs by plotting them
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # Plot the complete barbell graph
    nx.draw(G_complete, ax=axes[0], with_labels=True, node_size=500)
    axes[0].set_title('Complete Barbell Graph')
    # Plot the barbell graph with the bridge removed
    # nx.draw(G_single_edge, ax=axes[1], with_labels=True, node_size=500)
    # axes[1].set_title('Barbell Graph with Bridge Removed')
    # Plot the barbell graph with the bell connection removed

    plt.tight_layout()
    plt.show()

    # add suffixes to nodes
    G_complete = nx.relabel_nodes(G_complete, lambda x: str(x) + '.p')
    G_complete_eigs = laplacian_eigendecomp(weighted_laplacian_matrix(G_complete))
    diff_kernel_complete = [G_complete_eigs[1] @ np.diag(np.exp(-t * G_complete_eigs[0])) @ G_complete_eigs[1].T for t in t_values]

    all_gdds = {}
    diff_kernels_allknocks = []
    gdd_values_allknocks = []

    # knock down each node, calculate GDDs and plot
    for node in G_complete.nodes():
        node_base = node.rstrip('.p')
        knock_graph, knock_lap = knockdown_node_single_layer(G_complete, node_base, '.p', reduced_weight=0.05)

        knock_graph_eigs = laplacian_eigendecomp(knock_lap)
        diff_kernel_knock = [knock_graph_eigs[1] @ np.diag(np.exp(-t * knock_graph_eigs[0])) @ knock_graph_eigs[1].T for t in t_values]
        diff_kernels_allknocks.append(diff_kernel_knock)

        gdd_values_disruptA = np.linalg.norm(np.array(diff_kernel_complete) - np.array(diff_kernel_knock), axis=(1, 2), ord='fro')**2
        gdd_values_allknocks.append(gdd_values_disruptA)

        # if node_base == '8' or node_base == '5':
        #     print(knock_lap)
        #     print('\n')

        all_gdds[node_base] = gdd_values_disruptA

    # # check if the diff_kernels in diff_kernels_allknocks are the same as each other
    # for i in range(len(diff_kernels_allknocks)):
    #     for j in range(len(diff_kernels_allknocks)):
    #         if i != j:
    #             print(f'Are the diff_kernels for node {i} and node {j} the same? {np.allclose(diff_kernels_allknocks[i], diff_kernels_allknocks[j])}')

    # # check if the gdd_values in gdd_values_allknocks are the same as each other
    # for i in range(len(gdd_values_allknocks)):
    #     for j in range(len(gdd_values_allknocks)):
    #         if i != j:
    #             print(f'Are the gdd_values for node {i} and node {j} the same? {np.allclose(gdd_values_allknocks[i], gdd_values_allknocks[j])}')

    # # get the max GDD for each node
    # max_gdds = {}
    # for node_base in all_gdds.keys():
    #     max_gdds[node_base] = np.max(np.sqrt((all_gdds[node_base])))
    #     print(f'Max GDD for node {node_base}: {max_gdds[node_base]}')

    plt.figure(figsize=(10, 6))

    for node_base in all_gdds.keys():

        if node_base != '2' and node_base != '5':
            continue
        # plot the GDDs

        plt.plot(t_values, all_gdds[node_base], label=f'Node {node_base}', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('GDD Value')
        plt.title(f'GDD Values for {node_base} Knockdown')
        plt.legend()
        plt.grid(True)

        max_gdd = np.max(np.sqrt(all_gdds[node_base]))
        print(f'Max GDD for node {node_base}: {max_gdd}')


    plt.show()

    # %%
    def compare_nodes(m1, node_to_compare):
        G_complete = nx.barbell_graph(m1, 0)
        # ... rest of the graph generation and modification logic

        # Your existing logic for generating diff_kernel_complete and all_gdds
        # ...

        # Plot GDD values for node '2' and node_to_compare
        plt.figure(figsize=(10, 6))
        plt.plot(t_values, all_gdds['2'], label='Node 2', alpha=0.5)
        plt.plot(t_values, all_gdds[node_to_compare], label=f'Node {node_to_compare}', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('GDD Value')
        plt.title(f'GDD Values for Node 2 vs Node {node_to_compare} Knockdown')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Main loop for varying m1
    for m1 in range(5, 35, 5):
        node_to_compare = str(m1 * 2)
        compare_nodes(m1, node_to_compare)









    ###
    # %%
    # %% ############################ BARBELL GRAPHS ############################
    # The first graph will be a complete barbell graph, the second will have its bridge removed,
    # and the third will have its central connection removed.


    t_values = np.linspace(0, 10, 500)


    # Define the number of nodes in the barbell graph's complete subgraphs and the bridge length
    m1 = 5  # Number of nodes in the complete subgraphs
    m2 = 0  # Number of nodes in the bridge


    # Generate the complete barbell graph
    G_single_edge = nx.barbell_graph(m1, m2)


    # Identify the nodes to move and disconnect
    node_to_move_from_first_bell = m1 - 2  # Second to last node in the first bell
    node_to_move_from_second_bell = m1 + 1  # Second node in the second bell


    G_complete = G_single_edge.copy()
    # Add the new edge directly connecting the two identified nodes
    # G_complete.add_edge(node_to_move_from_first_bell, node_to_move_from_second_bell)


    # Verify the graphs by plotting them
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # Plot the complete barbell graph
    nx.draw(G_complete, ax=axes[0], with_labels=True, node_size=500)
    axes[0].set_title('Complete Barbell Graph')
    # Plot the barbell graph with the bridge removed
    # nx.draw(G_single_edge, ax=axes[1], with_labels=True, node_size=500)
    # axes[1].set_title('Barbell Graph with Bridge Removed')
    # Plot the barbell graph with the bell connection removed


    plt.tight_layout()
    plt.show()


    # add suffixes to nodes
    G_complete = nx.relabel_nodes(G_complete, lambda x: str(x) + '.p')
    G_complete_eigs = laplacian_eigendecomp(weighted_laplacian_matrix(G_complete))
    diff_kernel_complete = [G_complete_eigs[1] @ np.diag(np.exp(-t * G_complete_eigs[0])) @ G_complete_eigs[1].T for t in t_values]


    all_gdds = {}
    diff_kernels_allknocks = []
    gdd_values_allknocks = []


    # knock down each node, calculate GDDs and plot
    for node in G_complete.nodes():
        node_base = node.rstrip('.p')
        knock_graph, knock_lap = knockdown_node_single_layer(G_complete, node_base, '.p', reduced_weight=0.05)


        knock_graph_eigs = laplacian_eigendecomp(knock_lap)
        diff_kernel_knock = [knock_graph_eigs[1] @ np.diag(np.exp(-t * knock_graph_eigs[0])) @ knock_graph_eigs[1].T for t in t_values]
        diff_kernels_allknocks.append(diff_kernel_knock)


        gdd_values_disruptA = np.linalg.norm(np.array(diff_kernel_complete) - np.array(diff_kernel_knock), axis=(1, 2), ord='fro')**2
        gdd_values_allknocks.append(gdd_values_disruptA)


        # if node_base == '8' or node_base == '5':
        #     print(knock_lap)
        #     print('\n')


        all_gdds[node_base] = gdd_values_disruptA


    # # check if the diff_kernels in diff_kernels_allknocks are the same as each other
    # for i in range(len(diff_kernels_allknocks)):
    #     for j in range(len(diff_kernels_allknocks)):
    #         if i != j:
    #             print(f'Are the diff_kernels for node {i} and node {j} the same? {np.allclose(diff_kernels_allknocks[i], diff_kernels_allknocks[j])}')


    # # check if the gdd_values in gdd_values_allknocks are the same as each other
    # for i in range(len(gdd_values_allknocks)):
    #     for j in range(len(gdd_values_allknocks)):
    #         if i != j:
    #             print(f'Are the gdd_values for node {i} and node {j} the same? {np.allclose(gdd_values_allknocks[i], gdd_values_allknocks[j])}')


    # # get the max GDD for each node
    # max_gdds = {}
    # for node_base in all_gdds.keys():
    #     max_gdds[node_base] = np.max(np.sqrt((all_gdds[node_base])))
    #     print(f'Max GDD for node {node_base}: {max_gdds[node_base]}')


    plt.figure(figsize=(10, 6))


    for node_base in all_gdds.keys():


        if node_base != '2' and node_base != '5':
            continue
        # plot the GDDs


        plt.plot(t_values, all_gdds[node_base], label=f'Node {node_base}', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('GDD Value')
        plt.title(f'GDD Values for {node_base} Knockdown')
        plt.legend()
        plt.grid(True)


        max_gdd = np.max(np.sqrt(all_gdds[node_base]))
        print(f'Max GDD for node {node_base}: {max_gdd}')




    plt.show()


    # %%
    def compare_nodes(m1, node_to_compare):
        G_complete = nx.barbell_graph(m1, 0)
        # ... rest of the graph generation and modification logic


        # Your existing logic for generating diff_kernel_complete and all_gdds
        # ...


        # Plot GDD values for node '2' and node_to_compare
        plt.figure(figsize=(10, 6))
        plt.plot(t_values, all_gdds['2'], label='Node 2', alpha=0.5)
        plt.plot(t_values, all_gdds[node_to_compare], label=f'Node {node_to_compare}', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('GDD Value')
        plt.title(f'GDD Values for Node 2 vs Node {node_to_compare} Knockdown')
        plt.legend()
        plt.grid(True)
        plt.show()


    # Main loop for varying m1
    for m1 in range(5, 35, 5):
        node_to_compare = str(m1 * 2)
        compare_nodes(m1, node_to_compare)








    # %%
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_graph_with_weighted_edges(G):
        pos = nx.spring_layout(G)  # or any other layout algorithm you prefer

        # Compute the maximum weight in the graph for normalization
        max_weight = max([G[u][v]['weight'] for u, v in G.edges()])

        # Draw nodes
        nx.draw_networkx_nodes(G, pos)

        # Draw edges with alpha representing the weight
        for u, v in G.edges():
            weight = G[u][v]['weight']
            alpha = weight / max_weight  # Normalizing alpha values
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=alpha)

        # Draw labels
        nx.draw_networkx_labels(G, pos)

        plt.show()

    max_gdd_differences = []
    m1_values = []

    m2 = 0

    # Main loop for varying m1
    for m1 in range(5, 30, 5):
        node_to_compare = str(m1)
        # G_complete = nx.barbell_graph(m1, 0)
        # Generate the complete barbell graph
        G_single_edge = nx.barbell_graph(m1, m2)

        # Identify the nodes to move and disconnect
        node_to_move_from_first_bell = m1 - 2  # Second to last node in the first bell
        node_to_move_from_second_bell = m1 + 1  # Second node in the second bell

        G_complete = G_single_edge.copy()
        # Add the new edge directly connecting the two identified nodes
        G_complete.add_edge(node_to_move_from_first_bell, node_to_move_from_second_bell)


        G_complete = nx.relabel_nodes(G_complete, lambda x: str(x) + '.p')
        G_complete_eigs = laplacian_eigendecomp(weighted_laplacian_matrix(G_complete))
        diff_kernel_complete = [G_complete_eigs[1] @ np.diag(np.exp(-t * G_complete_eigs[0])) @ G_complete_eigs[1].T for t in t_values]


        max_gdds = {}


        for node in ['2.p', f'{node_to_compare}.p']:
            node_base = node.rstrip('.p')
            knock_graph, knock_lap = knockdown_node_single_layer(G_complete, node_base, '.p', reduced_weight=0.00)

            # plot_graph_with_weighted_edges(knock_graph)

            knock_graph_eigs = laplacian_eigendecomp(knock_lap)
            diff_kernel_knock = [knock_graph_eigs[1] @ np.diag(np.exp(-t * knock_graph_eigs[0])) @ knock_graph_eigs[1].T for t in t_values]


            gdd_values_disrupt = np.linalg.norm(np.array(diff_kernel_complete) - np.array(diff_kernel_knock), axis=(1, 2), ord='fro')**2
            max_gdd = np.max(np.sqrt(gdd_values_disrupt)) / m1
            # print node and max_gdd
            print(f'Max GDD for node {node_base}: {max_gdd}')
            max_gdds[node_base] = max_gdd


        # Calculate and store the absolute difference of max GDD values
        max_gdd_diff = max_gdds[node_to_compare.rstrip('.p')] / max_gdds['2']  
        # print degree of node_to_compare
        # print(f'Degree of node {node_to_compare}: {G_complete.degree(node_to_compare)}')
        # print deger of node 2
        # print(f'Degree of node 2: {G_complete.degree("2.p")}')
        max_gdd_differences.append(max_gdd_diff)
        m1_values.append(m1)


    # Plotting the differences against m1 values
    plt.figure(figsize=(10, 6))
    plt.plot(m1_values, max_gdd_differences, marker='o')
    plt.xlabel('m1 value')
    plt.ylabel('Absolute Difference in Max GDD')
    plt.title('Absolute Difference in Max GDD for Node 2 vs Node m1')
    plt.grid(True)
    plt.show()


    print(max_gdd_differences)



    # %% PLOTTING THE MAX GDD
    # # Compute the Laplacian Matrices for both graphs
    # laplacian_A = weighted_laplacian_matrix(G_complete)
    # laplacian_B = weighted_laplacian_matrix(G_cc)

    # # Step 3 and 4: Compute the Laplacian Kernels and calculate the xi values over a range of t values
    # t_values = np.linspace(0, 10, 1000)
    # xi_values = []

    # for t in t_values:
    #     kernel_A = laplacian_exponential_kernel_eigendecomp(*np.linalg.eigh(laplacian_A), t)
    #     kernel_B = laplacian_exponential_kernel_eigendecomp(*np.linalg.eigh(laplacian_B), t)
    #     xi = np.linalg.norm((kernel_A - kernel_B), 'fro')
    #     xi_values.append(xi)

    # # Find the maximum xi value and the corresponding t value using line search
    # max_xi = max(xi_values)
    # max_xi_index = xi_values.index(max_xi)
    # max_xi_time = t_values[max_xi_index]

    # # Step 5: Plot xi against t
    # plt.figure(figsize=(10, 6))
    # plt.plot(t_values, xi_values, label='xi(t)')
    # plt.plot(max_xi_time, max_xi, 'ro', label='Max xi')
    # # add a text label at maximum point
    # plt.annotate(f'Max xi = {round(max_xi, 2)}\n at t = {round(max_xi_time, 2)}', xy=(max_xi_time, max_xi), xytext=(max_xi_time + 1, max_xi - 0.1),
    #              )
    # plt.xlabel('Diffusion Time (t)')
    # plt.ylabel('Xi Value (Graph Difference)')
    # plt.title('Xi Values Over Diffusion Time for Two Graphs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # %% TESTING EIGENDECOMP
    # # TESTING ON SCALE_FREE GRAPH WITH RANDOM WEIGHTS
    # weighted_laplacian = weighted_laplacian_matrix(weighted_scale_free_graph)

    # # Eigen-decomposition of the Laplacian matrix
    # eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    # # Test the eigen-decomposition approach for a single t value
    # t_test = 0.5
    # kernel_eigendecomp = laplacian_exponential_kernel_eigendecomp(eigenvalues, eigenvectors, t_test)
    # # Compare with the direct computation for verification
    # kernel_direct = laplacian_exponential_diffusion_kernel(laplacian_matrix, t_test)
    # # Check if the results are similar (within a small numerical tolerance)
    # np.allclose(kernel_eigendecomp, kernel_direct)

    # # Test the eigen-decomposition approach with the weighted Laplacian matrix
    # eigenvalues_weighted, eigenvectors_weighted = np.linalg.eigh(weighted_laplacian)
    # kernel_eigendecomp_weighted = laplacian_exponential_kernel_eigendecomp(eigenvalues_weighted, eigenvectors_weighted, t_test)
    # # Output the first few elements of the weighted Laplacian matrix and the kernel as a sanity check
    # print(f'weighted laplacian:\n {weighted_laplacian[:5, :5]}')
    # print(f'Reconstructed Kernel from eigen-decomposition (weighted):\n {kernel_eigendecomp_weighted[:5, :5]}')

    # # np.allclose(kernel_eigendecomp, kernel_direct)

    # %%
    # %%
    ################################################################# GRAPH PARAMETERS
    # N = 100  # Number of nodes
    # m = 3    # Number of edges to attach from a new node to existing nodes



    # # SCALE FREE GRAPH
    # scale_free_graph = nx.barabasi_albert_graph(N, m, seed=rand_seed)
    # laplacian_matrix = nx.laplacian_matrix(scale_free_graph).toarray()
    # # Assign random weights to each edge (for example, weights between 0.1 and 1.0)
    # weighted_scale_free_graph = scale_free_graph.copy()
    # for u, v in weighted_scale_free_graph.edges():
    #     weighted_scale_free_graph[u][v]['weight'] = np.random.uniform(0.1, 1.0)

    # weighted_split_scalefree_g, set_1, set_2 = adjust_inter_set_edge_weights(weighted_scale_free_graph, new_weight=0.01)

    # # get hub nodes
    # degree_dict = dict(scale_free_graph.degree(scale_free_graph.nodes()))
    # # get 3 nodes with largest degree
    # hub_nodes = sorted(degree_dict, key=lambda x: degree_dict[x], reverse=True)[:3]
    # low_nodes = sorted(degree_dict, key=lambda x: degree_dict[x])[:3]
    # print(f'hub nodes: {hub_nodes}')
    # print(f'anti-hubs nodes: {low_nodes}')


    # # RANDOM GRAPH
    # random_graph = nx.erdos_renyi_graph(N, 0.5, seed=rand_seed) 
    # random_laplacian = nx.laplacian_matrix(random_graph).toarray()
    # weighted_random_graph = random_graph.copy()
    # for u, v in weighted_random_graph.edges():
    #     weighted_random_graph[u][v]['weight'] = 1.0


    # %%
    # %%
    ################################################################# GRAPH PARAMETERS
    # N = 100  # Number of nodes
    # m = 3    # Number of edges to attach from a new node to existing nodes



    # # SCALE FREE GRAPH
    # scale_free_graph = nx.barabasi_albert_graph(N, m, seed=rand_seed)
    # laplacian_matrix = nx.laplacian_matrix(scale_free_graph).toarray()
    # # Assign random weights to each edge (for example, weights between 0.1 and 1.0)
    # weighted_scale_free_graph = scale_free_graph.copy()
    # for u, v in weighted_scale_free_graph.edges():
    #     weighted_scale_free_graph[u][v]['weight'] = np.random.uniform(0.1, 1.0)

    # weighted_split_scalefree_g, set_1, set_2 = adjust_inter_set_edge_weights(weighted_scale_free_graph, new_weight=0.01)

    # # get hub nodes
    # degree_dict = dict(scale_free_graph.degree(scale_free_graph.nodes()))
    # # get 3 nodes with largest degree
    # hub_nodes = sorted(degree_dict, key=lambda x: degree_dict[x], reverse=True)[:3]
    # low_nodes = sorted(degree_dict, key=lambda x: degree_dict[x])[:3]
    # print(f'hub nodes: {hub_nodes}')
    # print(f'anti-hubs nodes: {low_nodes}')


    # # RANDOM GRAPH
    # random_graph = nx.erdos_renyi_graph(N, 0.5, seed=rand_seed) 
    # random_laplacian = nx.laplacian_matrix(random_graph).toarray()
    # weighted_random_graph = random_graph.copy()
    # for u, v in weighted_random_graph.edges():
    #     weighted_random_graph[u][v]['weight'] = 1.0
