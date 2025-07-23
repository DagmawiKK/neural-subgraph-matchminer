import argparse
import csv
from itertools import combinations
import time
import os
import pickle

from deepsnap.batch import Batch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder
from subgraph_mining.search_agents import GreedySearchAgent, MCTSSearchAgent, MemoryEfficientMCTSAgent, MemoryEfficientGreedyAgent, BeamSearchAgent

import matplotlib.pyplot as plt

import random
from scipy.io import mmread
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA

# def process_large_graph_in_chunks(graph, chunk_size=10000):
#     graph_chunks = []
#     # comment: could be optimized

def bfs_chunk(graph, start_node, max_size):
    visited = set([start_node])
    queue = [start_node]
    while queue and len(visited) < max_size:
        node = queue.pop(0)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= max_size:
                    break
    return graph.subgraph(visited).copy()

def process_large_graph_in_chunks(graph, chunk_size=10000):
    all_nodes = set(graph.nodes())
    graph_chunks = []
    while all_nodes:
        start_node = next(iter(all_nodes))
        chunk = bfs_chunk(graph, start_node, chunk_size)
        graph_chunks.append(chunk)
        all_nodes -= set(chunk.nodes())
    return graph_chunks

    # all_nodes = list(graph.nodes())
    
    # for i in range(0, len(all_nodes), chunk_size):
    #     chunk_nodes = all_nodes[i:i+chunk_size]
        
    #     # chunk_graph = graph.subgraph(chunk_nodes) remove not used
        
    #     extended_nodes = set(chunk_nodes)
    #     for node in chunk_nodes:
    #         neighbors = list(graph.neighbors(node))
    #         for neighbor in neighbors:
    #             if neighbor not in extended_nodes:
    #                 extended_nodes.add(neighbor)
    #                 if len(extended_nodes) > chunk_size * 1.2:  
    #                     break
        
    #     chunk_subgraph = graph.subgraph(extended_nodes)
    #     graph_chunks.append(chunk_subgraph)
    
    # return graph_chunks

def make_plant_dataset(size):
    generator = combined_syn.get_generator([size])
    random.seed(3001)
    np.random.seed(14853)
    pattern = generator.generate(size=10)
    nx.draw(pattern, with_labels=True)
    plt.savefig("plots/cluster/plant-pattern.png")
    plt.close()
    graphs = []
    for i in range(1000):
        graph = generator.generate()
        n_old = len(graph)
        graph = nx.disjoint_union(graph, pattern)
        for j in range(1, 3):
            u = random.randint(0, n_old - 1)
            v = random.randint(n_old, len(graph) - 1)
            graph.add_edge(u, v)
        graphs.append(graph)
    return graphs

def _process_chunk(args_tuple):
    chunk_dataset, task, args, chunk_index, total_chunks = args_tuple
    start_time = time.time()
    last_print = start_time
    print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} started chunk {chunk_index+1}/{total_chunks}", flush=True)
    try:
        # Simulate progress reporting every 10 seconds
        # If pattern_growth is long-running, you can add prints inside it as well
        result = None
        while result is None:
            now = time.time()
            if now - last_print >= 10:
                print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} still processing chunk {chunk_index+1}/{total_chunks} ({int(now-start_time)}s elapsed)", flush=True)
                last_print = now
            # Try to run pattern_growth, break if done
            result = pattern_growth([chunk_dataset], task, args)
        print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} finished chunk {chunk_index+1}/{total_chunks} in {int(time.time()-start_time)}s", flush=True)
        return result
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {e}", flush=True)
        return []

def pattern_growth_streaming(dataset, task, args):
    graph = dataset[0]
    graph_chunks = process_large_graph_in_chunks(graph, chunk_size=args.chunk_size)
    dataset = graph_chunks

    all_discovered_patterns = []

    total_chunks = len(dataset)
    chunk_args = [(chunk_dataset, task, args, idx, total_chunks) for idx, chunk_dataset in enumerate(dataset)]

    with mp.Pool(processes=4) as pool:
        results = pool.map(_process_chunk, chunk_args)

    for chunk_out_graphs in results:
        if chunk_out_graphs:
            all_discovered_patterns.extend(chunk_out_graphs)

    return all_discovered_patterns

def pattern_growth(dataset, task, args):
    start_time = time.time()
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path,
        map_location=utils.get_device()))

    if task == "graph-labeled":
        dataset, labels = dataset

    neighs_pyg, neighs = [], []
    print(len(dataset), "graphs")
    print("search strategy:", args.search_strategy)
    if task == "graph-labeled": print("using label 0")
    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0: continue
        if task == "graph-truncate" and i >= 1000: break
        if not type(graph) == nx.Graph:
            graph = pyg_utils.to_networkx(graph).to_undirected()
            for node in graph.nodes():
                if 'label' not in graph.nodes[node]:
                    graph.nodes[node]['label'] = str(node)
                if 'id' not in graph.nodes[node]:
                    graph.nodes[node]['id'] = str(node)
        graphs.append(graph)
    
    if args.use_whole_graphs:
        neighs = graphs
    else:
        anchors = []
        if args.sample_method == "radial":
            for i, graph in enumerate(graphs):
                print(i)
                for j, node in enumerate(graph.nodes):
                    if len(dataset) <= 10 and j % 100 == 0: print(i, j)
                    if args.use_whole_graphs:
                        neigh = graph.nodes
                    else:
                        neigh = list(nx.single_source_shortest_path_length(graph,
                            node, cutoff=args.radius).keys())
                        if args.subgraph_sample_size != 0:
                            neigh = random.sample(neigh, min(len(neigh),
                                args.subgraph_sample_size))
                    if len(neigh) > 1:
                        subgraph = graph.subgraph(neigh)
                        if args.subgraph_sample_size != 0:
                            subgraph = subgraph.subgraph(max(
                                nx.connected_components(subgraph), key=len))
                        
                        orig_attrs = {n: subgraph.nodes[n].copy() for n in subgraph.nodes()}
                        edge_attrs = {(u,v): subgraph.edges[u,v].copy() 
                                    for u,v in subgraph.edges()}
                        
                        mapping = {old: new for new, old in enumerate(subgraph.nodes())}
                        subgraph = nx.relabel_nodes(subgraph, mapping)
                        
                        for old, new in mapping.items():
                            subgraph.nodes[new].update(orig_attrs[old])
                        
                        for (old_u, old_v), attrs in edge_attrs.items():
                            subgraph.edges[mapping[old_u], mapping[old_v]].update(attrs)
                        
                        subgraph.add_edge(0, 0)
                        subgraph.nodes[0]["anchor"] = True
                        neighs.append(subgraph)
                        if args.node_anchored:
                            anchors.append(0)
        elif args.sample_method == "tree":
            start_time = time.time()
            for j in tqdm(range(args.n_neighborhoods)):
                graph, neigh = utils.sample_neigh(graphs,
                    random.randint(args.min_neighborhood_size,
                        args.max_neighborhood_size))
                neigh = graph.subgraph(neigh)
                neigh = nx.convert_node_labels_to_integers(neigh)
                neigh.add_edge(0, 0)
                neigh.nodes[0]["anchor"] = True
                neighs.append(neigh)
                if args.node_anchored:
                    anchors.append(0)

    embs = []
    if len(neighs) % args.batch_size != 0:
        print("WARNING: number of graphs not multiple of batch size")
    for i in range(len(neighs) // args.batch_size):
        top = (i+1)*args.batch_size
        with torch.no_grad():
            batch = utils.batch_nx_graphs(neighs[i*args.batch_size:top],
                anchors=anchors if args.node_anchored else None)
            emb = model.emb_model(batch)
            emb = emb.to(torch.device("cpu"))
        embs.append(emb)

    if args.analyze:
        embs_np = torch.stack(embs).numpy()
        plt.scatter(embs_np[:,0], embs_np[:,1], label="node neighborhood")

    if not hasattr(args, 'n_workers'):
        args.n_workers = mp.cpu_count()

    if args.search_strategy == "mcts":
        assert args.method_type == "order"
        if args.memory_efficient:
            agent = MemoryEfficientMCTSAgent(args.min_pattern_size, args.max_pattern_size,
                model, graphs, embs, node_anchored=args.node_anchored,
                analyze=args.analyze, out_batch_size=args.out_batch_size)
        else:
            agent = MCTSSearchAgent(args.min_pattern_size, args.max_pattern_size,
                model, graphs, embs, node_anchored=args.node_anchored,
                analyze=args.analyze, out_batch_size=args.out_batch_size)
    elif args.search_strategy == "greedy":
        if args.memory_efficient:
            agent = MemoryEfficientGreedyAgent(args.min_pattern_size, args.max_pattern_size,
                model, graphs, embs, node_anchored=args.node_anchored,
                analyze=args.analyze, model_type=args.method_type,
                out_batch_size=args.out_batch_size)
        else:
            agent = GreedySearchAgent(args.min_pattern_size, args.max_pattern_size,
                model, graphs, embs, node_anchored=args.node_anchored,
                analyze=args.analyze, model_type=args.method_type,
                out_batch_size=args.out_batch_size, n_beams=1,
                n_workers=args.n_workers)
        agent.args = args
    elif args.search_strategy == "beam":
        agent = BeamSearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size, beam_width=args.beam_width)
    out_graphs = agent.run_search(args.n_trials)
    
    print(time.time() - start_time, "TOTAL TIME")
    x = int(time.time() - start_time)
    print(x // 60, "mins", x % 60, "secs")

    count_by_size = defaultdict(int)
    for pattern in out_graphs:
            try:
                plt.figure(figsize=(15, 10))  
        
                node_labels = {}
                for n in pattern.nodes():
                    node_id = pattern.nodes[n].get('id', str(n))
                    node_label = pattern.nodes[n].get('label', 'unknown')
                    node_labels[n] = f"{node_id}:\n{node_label}"
        
                pos = nx.spring_layout(pattern, k=2.0, seed=42, iterations=50)
        
                unique_labels = sorted(set(pattern.nodes[n].get('label', 'unknown') for n in pattern.nodes()))
                label_color_map = {label: plt.cm.Set3(i) for i, label in enumerate(unique_labels)}
        
                colors = []
                node_sizes = []
                node_shapes = []
                for i, node in enumerate(pattern.nodes()):
                    node_label = pattern.nodes[node].get('label', 'unknown')
                    
                    if args.node_anchored and pattern.nodes[node].get("anchor", False):
                        colors.append('red')
                        node_sizes.append(5000)
                        node_shapes.append('s')  
                    else:
                        colors.append(label_color_map[node_label])
                        node_sizes.append(3000)
                        node_shapes.append('o')  
        
                for shape in set(node_shapes):
                    shape_indices = [i for i, s in enumerate(node_shapes) if s == shape]
                    nx.draw_networkx_nodes(pattern, pos, 
                                    nodelist=[list(pattern.nodes())[i] for i in shape_indices],
                                    node_color=[colors[i] for i in shape_indices], 
                                    node_size=[node_sizes[i] for i in shape_indices], 
                                    node_shape=shape,
                                    edgecolors='black', 
                                    linewidths=1.5)
        
                nx.draw_networkx_edges(pattern, pos, 
                                width=2,  
                                edge_color='gray',  
                                alpha=0.7)  
        
                nx.draw_networkx_labels(pattern, pos, 
                                 labels=node_labels, 
                                 font_size=9, 
                                 font_weight='bold',
                                 font_color='black',
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
                edge_labels = {(u,v): data.get('type', '') 
                        for u,v,data in pattern.edges(data=True)}
                nx.draw_networkx_edge_labels(pattern, pos, 
                                      edge_labels=edge_labels, 
                                      font_size=8, 
                                      font_color='darkred',  
                                      bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
                plt.title(f"Pattern Graph (Size: {len(pattern)} nodes)")
                plt.axis('off')  
        
                pattern_info = [f"{len(pattern)}-{count_by_size[len(pattern)]}"]
        
                node_types = sorted(set(pattern.nodes[n].get('label', '') for n in pattern.nodes()))
                if any(node_types):
                    pattern_info.append('nodes-' + '-'.join(node_types))
            
                edge_types = sorted(set(pattern.edges[e].get('type', '') for e in pattern.edges()))
                if any(edge_types):
                    pattern_info.append('edges-' + '-'.join(edge_types))
        
                filename = '_'.join(pattern_info)
                plt.tight_layout()
                plt.savefig(f"plots/cluster/{filename}.png", bbox_inches='tight', dpi=300)
                plt.savefig(f"plots/cluster/{filename}.pdf", bbox_inches='tight')
                plt.close()
                count_by_size[len(pattern)] += 1
        
            except Exception as e:
                print(f"Error visualizing pattern graph: {e}")
                continue

    if not os.path.exists("results"):
        os.makedirs("results")
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f)
    
    return out_graphs

def main():
    if not os.path.exists("plots/cluster"):
        os.makedirs("plots/cluster")

    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    
    args = parser.parse_args()

    print("Using dataset {}".format(args.dataset))
    if args.dataset.endswith('.pkl'):
        with open(args.dataset, 'rb') as f:
            data = pickle.load(f)
            graph = nx.Graph()
            graph.add_nodes_from(data['nodes'])
            graph.add_edges_from(data['edges'])
        dataset = [graph]
        task = 'graph'
        print(f"Loaded Networkx graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    elif args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'
    elif args.dataset == 'cox2':
        dataset = TUDataset(root='/tmp/cox2', name='COX2')
        task = 'graph'
    elif args.dataset == 'reddit-binary':
        dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
        task = 'graph'
    elif args.dataset == 'dblp':
        dataset = TUDataset(root='/tmp/dblp', name='DBLP_v1')
        task = 'graph-truncate '
    elif args.dataset == 'coil':
        dataset = TUDataset(root='/tmp/coil', name='COIL-DEL')
        task = 'graph'
    elif args.dataset.startswith('roadnet-'):
        graph = nx.Graph()
        with open("data/{}.txt".format(args.dataset), "r") as f:
            for row in f:
                if not row.startswith("#"):
                    a, b = row.split("\t")
                    graph.add_edge(int(a), int(b))
        dataset = [graph]
        task = 'graph'
    elif args.dataset == "ppi":
        dataset = PPI(root="/tmp/PPI")
        task = 'graph'
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
        task = 'graph'
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
        dataset = make_plant_dataset(size)
        task = 'graph'

    # if len(dataset) == 1 and dataset[0].number_of_nodes() > 100000:
    #     pattern_growth_streaming(dataset, task, args)
    # else:
    pattern_growth(dataset, task, args)

if __name__ == '__main__':
    main()