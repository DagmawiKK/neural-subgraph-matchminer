import argparse
import csv
import time
import os
import json
import concurrent.futures
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from subgraph_mining import decoder

from tqdm import tqdm
import matplotlib.pyplot as plt

from multiprocessing import Pool
import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA
from itertools import combinations

# import argparse
# from subgraph_mining.config import parse_decoder
# from subgraph_matching.config import parse_encoder

# parser = argparse.ArgumentParser(description='Decoder arguments')
# parse_decoder(parser)
# args = parser.parse_args()

# Increase timeout for large graphs
MAX_SEARCH_TIME = 1800  # 30 minutes for large graph processing
MAX_MATCHES_PER_QUERY = 10000
DEFAULT_SAMPLE_ANCHORS = 1000
CHECKPOINT_INTERVAL = 100  # Save progress every 100 tasks

def compute_graph_stats(G):
    """Compute graph statistics for filtering."""
    stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'degree_seq': sorted([d for _, d in G.degree()], reverse=True),
        'avg_degree': sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1)
    }
    
    # Add connected components info
    try:
        if args.graph_type == "directed":
            stats['n_components'] = nx.number_strongly_connected_components(G)
        else:
            stats['n_components'] = nx.number_connected_components(G)
    except:
        stats['n_components'] = 1  # Assume connected if there's an error
        
    return stats

def can_be_isomorphic(query_stats, target_stats):
    """Enhanced check if query could possibly be isomorphic to a subgraph of target."""
    # Basic size checks
    if query_stats['n_nodes'] > target_stats['n_nodes']:
        return False
    if query_stats['n_edges'] > target_stats['n_edges']:
        return False
    
    # More detailed checks
    # Check if query's max degree exceeds target's max degree
    if len(query_stats['degree_seq']) > 0 and len(target_stats['degree_seq']) > 0:
        if query_stats['degree_seq'][0] > target_stats['degree_seq'][0]:
            return False
    
    # Average degree comparison with tolerance
    if query_stats['avg_degree'] > target_stats['avg_degree'] * 1.1:  # 10% tolerance
        return False
    
    return True

def arg_parse():
    parser = argparse.ArgumentParser(description='count graphlets in a graph')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--queries_path', type=str)
    parser.add_argument('--graph_type', type=str, default="undirected")
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--count_method', type=str)
    parser.add_argument('--baseline', type=str)
    parser.add_argument('--node_anchored', action="store_true")
    parser.add_argument('--preserve_labels', action="store_true", help='Preserve node and edge labels during counting')
    parser.add_argument('--max_query_size', type=int, default=20, help='Maximum query size to process')
    parser.add_argument('--sample_anchors', type=int, default=DEFAULT_SAMPLE_ANCHORS, help='Number of anchor nodes to sample for large graphs')
    parser.add_argument('--checkpoint_file', type=str, default="checkpoint.json", help='File to save/load progress')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for processing')
    parser.add_argument('--timeout', type=int, default=MAX_SEARCH_TIME, help='Timeout per task in seconds')
    parser.add_argument('--use_sampling', action="store_true", help='Use node sampling for very large graphs')
    parser.set_defaults(dataset="enzymes",
                       queries_path="results/out-patterns.p",
                       out_path="results/counts.json",
                       n_workers=4,
                       count_method="bin",
                       baseline="none",
                       preserve_labels=False)
    return parser.parse_args()

def load_networkx_graph(filepath):
    """Load a Networkx graph from pickle format matching decoder's approach"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
        # If it's already a NetworkX graph, return it directly
        if isinstance(data, (nx.Graph, nx.DiGraph)):
            return data
            
        # If it's a PyG Data object, convert to NetworkX
        try:
            from torch_geometric.data import Data
            if isinstance(data, Data):
                return pyg_utils.to_networkx(data).to_undirected()
        except ImportError:
            pass
            
        # If it's a dictionary with graph data
        if isinstance(data, dict):
            if args.graph_type == "directed":
                graph = nx.DiGraph()
            else:
                graph = nx.Graph()
            
            # Handle node attributes
            if 'node_attrs' in data:
                for node_id, attrs in enumerate(data['node_attrs']):
                    graph.add_node(node_id, **attrs)
            elif 'node_features' in data:
                for node_id, feats in enumerate(data['node_features']):
                    graph.add_node(node_id, features=feats)
            
            # Handle edge indices
            if 'edge_index' in data:
                edge_index = data['edge_index']
                for src, dst in zip(edge_index[0], edge_index[1]):
                    graph.add_edge(src.item(), dst.item())
            
            return graph
            
        raise ValueError(f"Unknown pickle format in {filepath}")


def count_graphlets_helper(inp):
    """Worker function to count pattern occurrences with better timeout handling."""
    i, query, target, method, node_anchored, anchor_or_none, preserve_labels, timeout = inp
    
    start_time = time.time()
    
    # Set a maximum execution time - shorter than the given timeout
    effective_timeout = min(timeout, 600)  # Max 10 minutes per task
    
    # Quick stats check before proceeding
    query_stats = compute_graph_stats(query)
    target_stats = compute_graph_stats(target)
    if not can_be_isomorphic(query_stats, target_stats):
        return i, 0
    
    # Remove self loops
    query = query.copy()
    query.remove_edges_from(nx.selfloop_edges(query))
    target = target.copy()
    target.remove_edges_from(nx.selfloop_edges(target))

    count = 0
    try:
        # Use signal-based timeout to ensure we don't get stuck
        # This will only work on Unix-based systems
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Task {i} timed out after {effective_timeout} seconds")
            
        # Set the signal handler and a alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(effective_timeout)
        
        # Pre-compute query stats for method "freq"
        if method == "freq":
            ismags = nx.isomorphism.ISMAGS(query, query)
            n_symmetries = len(list(ismags.isomorphisms_iter(symmetry=False)))
        
        if method == "bin":
            if node_anchored:
                nx.set_node_attributes(target, 0, name="anchor")
                target.nodes[anchor_or_none]["anchor"] = 1
                
                if preserve_labels:
                    # Use lambda functions to properly match node and edge attributes
                    if args.graph_type == "directed":
                        matcher = iso.DiGraphMatcher(target, query,
                            node_match=lambda n1, n2: (n1.get("anchor") == n2.get("anchor") and
                                                    n1.get("label") == n2.get("label")),
                            edge_match=lambda e1, e2: e1.get("type") == e2.get("type"))
                    else:
                        matcher = iso.GraphMatcher(target, query,
                            node_match=lambda n1, n2: (n1.get("anchor") == n2.get("anchor") and
                                                    n1.get("label") == n2.get("label")),
                            edge_match=lambda e1, e2: e1.get("type") == e2.get("type"))
                else:
                    if args.graph_type == "directed":
                        matcher = iso.DiGraphMatcher(target, query,
                            node_match=iso.categorical_node_match(["anchor"], [0]))
                    else:
                        matcher = iso.GraphMatcher(target, query,
                            node_match=iso.categorical_node_match(["anchor"], [0]))
                
                if time.time() - start_time > timeout:
                    print(f"Timeout on query {i} before isomorphism check")
                    return i, 0
                
                # Perform isomorphism check
                count = int(matcher.subgraph_is_isomorphic())
            else:
                if preserve_labels:
                    matcher = iso.GraphMatcher(target, query,
                        node_match=lambda n1, n2: n1.get("label") == n2.get("label"),
                        edge_match=lambda e1, e2: e1.get("type") == e2.get("type"))
                else:
                    matcher = iso.GraphMatcher(target, query)
                
                if time.time() - start_time > timeout:
                    print(f"Timeout on query {i} before isomorphism check")
                    return i, 0
                
                count = int(matcher.subgraph_is_isomorphic())
        elif method == "freq":
            if preserve_labels:
                matcher = iso.GraphMatcher(target, query,
                    node_match=lambda n1, n2: n1.get("label") == n2.get("label"),
                    edge_match=lambda e1, e2: e1.get("type") == e2.get("type"))
            else:
                matcher = iso.GraphMatcher(target, query)
            
            count = 0
            for _ in matcher.subgraph_isomorphisms_iter():
                if time.time() - start_time > timeout:
                    print(f"Timeout during isomorphism iteration for query {i}")
                    break
                count += 1
                if count >= MAX_MATCHES_PER_QUERY:
                    break
            
            if method == "freq" and n_symmetries > 0:
                count = count / n_symmetries
        
        # Cancel the alarm
        signal.alarm(0)
            
    except TimeoutError as e:
        print(f"Task {i} timed out: {str(e)}")
        count = 0
    except Exception as e:
        print(f"Error processing query {i}: {str(e)}")
        count = 0
        
    processing_time = time.time() - start_time
    if processing_time > 10:  # Only log if it took significant time
        print(f"Query {i} processed in {processing_time:.2f} seconds with count {count}")
        
    return i, count

def save_checkpoint(n_matches, checkpoint_file):
    """Save current progress to checkpoint file."""
    with open(checkpoint_file, 'w') as f:
        json.dump({str(k): v for k, v in n_matches.items()}, f)
    print(f"Checkpoint saved to {checkpoint_file}")

def load_checkpoint(checkpoint_file):
    """Load progress from checkpoint file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            try:
                checkpoint = json.load(f)
                return defaultdict(float, {int(k): v for k, v in checkpoint.items()})
            except json.JSONDecodeError:
                print(f"Error loading checkpoint file {checkpoint_file}, starting fresh")
    return defaultdict(float)

def sample_subgraphs(target, n_samples=10, max_size=1000):
    """Sample manageable subgraphs from a very large graph."""
    subgraphs = []
    nodes = list(target.nodes())
    
    for _ in range(n_samples):
        # Start with a random node
        start_node = random.choice(nodes)
        subgraph_nodes = {start_node}
        if args.graph_type == "directed":
            frontier = list(target.successors(start_node))
        else:
            frontier = list(target.neighbors(start_node))
        
        # Grow the subgraph by BFS
        while len(subgraph_nodes) < max_size and frontier:
            next_node = frontier.pop(0)
            if next_node not in subgraph_nodes:
                subgraph_nodes.add(next_node)
                if args.graph_type == "directed":
                    frontier.extend([n for n in target.successors(next_node) 
                                    if n not in subgraph_nodes and n not in frontier])
                else:
                    frontier.extend([n for n in target.neighbors(next_node) 
                                    if n not in subgraph_nodes and n not in frontier])
        
        sg = target.subgraph(subgraph_nodes)
        subgraphs.append(sg)
        
    return subgraphs

def count_graphlets(queries, targets, args):
    """Count graph patterns with improved handling for large graphs."""
    print(f"Processing {len(queries)} queries across {len(targets)} targets")
    
    # Load checkpoint if exists
    n_matches = load_checkpoint(args.checkpoint_file)
    
    # Load or create problematic tasks list
    problematic_tasks_file = "problematic_tasks.json"
    if os.path.exists(problematic_tasks_file):
        with open(problematic_tasks_file, 'r') as f:
            try:
                problematic_tasks = set(json.load(f))
                print(f"Loaded {len(problematic_tasks)} problematic tasks to skip")
            except:
                problematic_tasks = set()
    else:
        problematic_tasks = set()
    
    # For very large graphs, consider sampling
    if args.use_sampling and any(t.number_of_nodes() > 100000 for t in targets):
        sampled_targets = []
        for target in targets:
            if target.number_of_nodes() > 100000:
                print(f"Sampling subgraphs from large graph with {target.number_of_nodes()} nodes")
                sampled_targets.extend(sample_subgraphs(target, n_samples=20, max_size=10000))
            else:
                sampled_targets.append(target)
        targets = sampled_targets
        print(f"After sampling: {len(targets)} target graphs to process")
    
    # Pre-compute graph statistics
    #target_stats = [compute_graph_stats(t) for t in targets]
    #query_stats = [compute_graph_stats(q) for q in queries]
    #changed to multiprocessing using 
    with Pool(processes=args.n_workers) as pool:

        target_stats = pool.map(compute_graph_stats, targets)
        
        query_stats = pool.map(compute_graph_stats, queries)
    
    # Generate work items with filtering
    inp = []
    for i, (query, q_stats) in enumerate(zip(queries, query_stats)):
        if query.number_of_nodes() > args.max_query_size:
            print(f"Skipping query {i}: exceeds max size {args.max_query_size}")
            continue
            
        for t_idx, (target, t_stats) in enumerate(zip(targets, target_stats)):
            # Skip if structures are incompatible
            if not can_be_isomorphic(q_stats, t_stats):
                continue
            
            task_id = f"{i}_{t_idx}"
            
            # Skip known problematic tasks
            if task_id in problematic_tasks:
                print(f"Skipping known problematic task {task_id}")
                continue
                
            if task_id in n_matches:
                print(f"Skipping already processed task {task_id}")
                continue
                
            if args.node_anchored:
                # Sample anchors for large graphs
                if target.number_of_nodes() > args.sample_anchors:
                    anchors = random.sample(list(target.nodes), args.sample_anchors)
                else:
                    anchors = list(target.nodes)
                    
                for anchor in anchors:
                    inp.append((i, query, target, args.count_method, args.node_anchored, anchor, 
                             args.preserve_labels, args.timeout))
            else:
                inp.append((i, query, target, args.count_method, args.node_anchored, None, 
                         args.preserve_labels, args.timeout))
    
    print(f"Generated {len(inp)} tasks after filtering")
    n_done = 0
    last_checkpoint = time.time()
   
    with Pool(processes=args.n_workers) as pool:
        for batch_start in range(0, len(inp), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(inp))
            batch = inp[batch_start:batch_end]

            print(f"Processing batch {batch_start}-{batch_end} out of {len(inp)}")
            batch_start_time = time.time()

            results = pool.imap_unordered(count_graphlets_helper, batch)

            for result in results:
                if time.time() - batch_start_time > 3600:  # 1-hour batch timeout
                    print(f"Batch {batch_start}-{batch_end} taking too long, marking remaining tasks problematic")
                    # Mark remaining tasks
                    for task in batch:
                        i = task[0]
                        task_id = f"{i}_{batch_start}"
                        problematic_tasks.add(task_id)
                    break

                i, n = result
                n_matches[i] += n
                n_done += 1

                if n_done % 10 == 0:
                    print(f"Processed {n_done}/{len(inp)} tasks, queries with matches: {sum(1 for v in n_matches.values() if v > 0)}/{len(n_matches)}", flush=True)

                # Periodic checkpoint save
                if time.time() - last_checkpoint > 300:
                    save_checkpoint(n_matches, args.checkpoint_file)
                    with open(problematic_tasks_file, 'w') as f:
                        json.dump(list(problematic_tasks), f)
                    last_checkpoint = time.time()

            # Save checkpoint after each batch
            save_checkpoint(n_matches, args.checkpoint_file)
            with open(problematic_tasks_file, 'w') as f:
                json.dump(list(problematic_tasks), f)

    print("\nDone counting")
    return [n_matches[i] for i in range(len(queries))]


#multiprocessing gen_baseline_queries ----------------
def generate_one_baseline(args):
    import networkx as nx
    import random

    i, query, targets, method = args

    if len(query) == 0:
        return query

    MAX_ATTEMPTS = 100  # Avoid infinite loops

    for attempt in range(MAX_ATTEMPTS):
        try:
            graph = random.choice(targets)
            if graph.number_of_nodes() == 0:
                continue

            if method == "radial":
                node = random.choice(list(graph.nodes))
                neigh = list(nx.single_source_shortest_path_length(graph, node, cutoff=3).keys())
                subgraph = graph.subgraph(neigh)
                if subgraph.number_of_nodes() == 0:
                    continue
                if args.graph_type == "directed":
                    largest_cc = max(nx.strongly_connected_components(subgraph), key=len)
                else:
                    largest_cc = max(nx.connected_components(subgraph), key=len)
                neigh = subgraph.subgraph(largest_cc)
                neigh = nx.convert_node_labels_to_integers(neigh)
                if len(neigh) == len(query):
                    return neigh

            elif method == "tree":
                start_node = random.choice(list(graph.nodes))
                neigh = [start_node]
                if args.graph_type == "directed":
                    frontier = list(set(graph.successors(start_node)) - set(neigh))
                else:
                    frontier = list(set(graph.neighbors(start_node)) - set(neigh))
                while len(neigh) < len(query) and frontier:
                    new_node = random.choice(frontier)
                    neigh.append(new_node)
                    if args.graph_type == "directed":
                        frontier += list(graph.successors(new_node))
                    else:
                        frontier += list(graph.neighbors(new_node))
                    frontier = [x for x in frontier if x not in neigh]
                if len(neigh) == len(query):
                    sub = graph.subgraph(neigh)
                    return nx.convert_node_labels_to_integers(sub)

        except Exception as e:
            continue  # Safe fallback on error

    print(f"[WARN] Baseline not found for query {i} after {MAX_ATTEMPTS} attempts.")
    return nx.Graph()  # Return empty graph if failed

def convert_to_networkx(graph):
    if isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph):
        return graph
    return pyg_utils.to_networkx(graph).to_undirected()
    
def gen_baseline_queries(queries, targets, method="radial", node_anchored=False):
    print(f"Generating {len(queries)} baseline queries in parallel using method: {method}")
    
    args_list = [(i, query, targets, method) for i, query in enumerate(queries)]
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(generate_one_baseline, args_list)
    
    return results


def main():
    global args
    args = arg_parse()
    print("Using {} workers".format(args.n_workers))
    print("Baseline:", args.baseline)
    print(f"Max query size: {args.max_query_size}")
    print(f"Timeout per task: {args.timeout} seconds")

    # Load dataset based on type
    if args.dataset.endswith('.pkl'):
        print(f"Loading Networkx graph from {args.dataset}")
        try:
            graph = load_networkx_graph(args.dataset)
           #print(f"Loaded Networkx graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            dataset = [graph]
        except Exception as e:
            print(f"Error loading graph: {str(e)}")
            raise
    elif args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    elif args.dataset == 'cox2':
        dataset = TUDataset(root='/tmp/cox2', name='COX2')
    elif args.dataset == 'reddit-binary':
        dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
    elif args.dataset == 'coil':
        dataset = TUDataset(root='/tmp/coil', name='COIL-DEL')
    elif args.dataset == 'ppi-pathways':
        graph = nx.Graph()
        with open("data/ppi-pathways.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                graph.add_edge(int(row[0]), int(row[1]))
        dataset = [graph]
    elif args.dataset in ['diseasome', 'usroads', 'mn-roads', 'infect']:
        fn = {"diseasome": "bio-diseasome.mtx",
            "usroads": "road-usroads.mtx",
            "mn-roads": "mn-roads.mtx",
            "infect": "infect-dublin.edges"}
        graph = nx.Graph()
        with open("data/{}".format(fn[args.dataset]), "r") as f:
            for line in f:
                if not line.strip(): continue
                a, b = line.strip().split(" ")
                graph.add_edge(int(a), int(b))
        dataset = [graph]
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
        dataset = decoder.make_plant_dataset(size)
    elif args.dataset == "analyze":
        with open("results/analyze.p", "rb") as f:
            cand_patterns, _ = pickle.load(f)
            queries = [q for score, q in cand_patterns[10]][:200]
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    #call convert to graph function
    with Pool(processes=args.n_workers) as pool:
        targets = pool.map(convert_to_networkx, dataset)

    # Load query patterns
    if args.dataset != "analyze":
        with open(args.queries_path, "rb") as f:
            queries = pickle.load(f)
            
    query_lens = [len(query) for query in queries]
    print(f"Loaded {len(queries)} query patterns")

    # Handle different counting methods
    if args.baseline == "exact":
        # Using exact counting for comparison
        print("Using exact counting method")
        n_matches = count_graphlets(queries, targets, args)
    elif args.baseline == "none":
        # Standard pattern counting
        n_matches = count_graphlets(queries, targets, args)
    else:
        # Generate baseline queries for comparison
        print(f"Generating baseline queries using {args.baseline}")
        baseline_queries = gen_baseline_queries(queries, targets,
            node_anchored=args.node_anchored, method=args.baseline)
        query_lens = [len(q) for q in baseline_queries]
        n_matches = count_graphlets(baseline_queries, targets, args)
            
    # Save results
    with open(args.out_path, "w") as f:
        json.dump((query_lens, n_matches, []), f)
    print(f"Results saved to {args.out_path}")



if __name__ == "__main__":
    main()