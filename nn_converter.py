#methods to convert a graph into the topology of a rudimentary neural network:
from trophic_web_methods import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import igraph as ig
import nvtx
import igraph as ig
from math import prod, ceil, floor
import numpy as np
from sklearn.mixture import GaussianMixture
from torch.amp import autocast, GradScaler
import logging


from scipy.special import softmax
from sklearn.metrics import pairwise_distances_argmin


#Straight from perplexity:
def baseline_fully_connected_network(
    n_in, n_out, E_target, min_layers=3, max_layers=10, max_iter=1000, lambda_layers=1.0
):
    min_width = min(n_in, n_out)
    best_loss = float('inf')
    best_config = None

    if n_in == 0 or n_out == 0:
        raise ValueError("Zero inputs / Zero outputs occurred")

    for L in range(min_layers, max_layers + 1):
        layer_sizes = [round(n_in + (i / (L - 1)) * (n_out - n_in)) for i in range(L)]
        for _ in range(max_iter):
            # Check the width constraint for hidden layers
            if any(h < min_width for h in layer_sizes[1:-1]):
                break  # Skip this configuration, does not meet width constraint
            edge_count = sum(layer_sizes[i] * layer_sizes[i + 1] for i in range(L - 1))
            diff = abs(edge_count - E_target)
            # Prefer more layers only if within 5% of edge target
            if diff <= 0.05 * E_target:
                loss = diff - lambda_layers * (L - 2)
            else:
                loss = diff
            if loss < best_loss:
                best_loss = loss
                best_config = layer_sizes.copy()
            if edge_count == E_target:
                break
            for i in range(1, L - 1):
                if edge_count > E_target and layer_sizes[i] > min_width:
                    layer_sizes[i] -= 1
                elif edge_count < E_target:
                    layer_sizes[i] += 1

    return best_config, best_loss





#Function to create a dense linearly scalable graph that imitates number of edges from a provided graph
def create_dense_graph(g_mimic, plot = False):

    g_mimic = g_mimic.copy()

    """
    Criteria:
    - Needs to have the same number of basal and apex species as the provided graph
    - Needs to have the same number of edges as the provided graph
    - Needs to be fully connected by layers
    - Needs to be DAG
    - Shouldnt have wild bottlenecks and ideally layer width follows the slope connecting count of start and end
    """
    #Compute the trophic levels of all nodes; set output layer to be the number of apex species; then, pop apex species from the graph

    #need to find approximate depth of the provided graph
    
    trophic_levels = trophic_level_indexer(g_mimic)

    #Find number of apex species
    apexes = [v.index for v in g_mimic.vs if g_mimic.degree(v.index, mode="out") == 0 and g_mimic.degree(v.index, mode="in") > 0] #apex
    num_apex = len(apexes)

    basals = [v.index for v in g_mimic.vs if g_mimic.degree(v.index, mode="in") == 0 and g_mimic.degree(v.index, mode="out") > 0] #basal
    num_basal = len(basals)

    total_connections = g_mimic.ecount()

    num_neurons, _ = baseline_fully_connected_network(num_basal, num_apex, total_connections, min_layers = 3, max_layers=(int(round(max(trophic_levels)))), lambda_layers=1)

        

    print(num_neurons)
    num_neurons = [int(round(x)) for x in num_neurons]


    #now, need to fully connect neurons:
    edges = []

    offset = 0

    for k in range(len(num_neurons) - 1):
        if k > 0:
            offset += num_neurons[k - 1]
            
        for i in range(offset, num_neurons[k] + offset):
            for j in range(offset + num_neurons[k], offset + num_neurons[k] + num_neurons[k + 1]):
                edges.append((i,j))
    

    g = ig.Graph(edges = edges, directed = True)

    return g, num_neurons


def horzcat_graph(g): #need a catch for when there's only one module, or just do extinction events once horzcat has happened
    #Goal:
    g = g.copy()
    #iterate through all edges
    #if an edge connects modules, add another edge with the same source node to a target of id target + num_nodes_total
    #then, add another edge with the same target node but with a source node of id source + num_nodes_total
    #add EVERY edge to a new array and add num_nodes_total to EVERY ID
    
    community = g.community_leading_eigenvector()
    membership = community.membership
    total_nodes = g.vcount()
    new_edges = []

    for edge in g.es:
        if membership[edge.source] != membership[edge.target]:
            #assuming source membership is a0 and target membership is b0,
            # add two edges:
            # a0 --> b1 #inter graph
            # a1 --> b0 #inter graph
            new_edges.append((edge.source + total_nodes, edge.target))
            new_edges.append((edge.source, edge.target + total_nodes))
        #then, copy remaining edges into new-sized graph
        new_edges.append((edge.source + total_nodes , edge.target + total_nodes))

    g.add_vertices(total_nodes)
    g.add_edges(new_edges)

    if "name" in g.vs.attributes():
        for i in range(total_nodes, total_nodes * 2):
            g.vs[i]["name"] = g.vs[i - total_nodes]["name"]
    #rename everything
    #ultimately retains same number of groups

    return g

def vertcat_graph(g1, g2): #assumes that the SAME type of graph is being stacked
    g1 = g1.copy()
    g2 = g2.copy()

    apexes = [v.index for v in g1.vs if g1.degree(v.index, mode="out") == 0 and g1.degree(v.index, mode="in") > 0] #apex
    num_apex = len(apexes)

    basals = [v.index for v in g2.vs if g2.degree(v.index, mode="in") == 0 and g2.degree(v.index, mode="out") > 0] #apex
    num_basal = len(basals)

    print(num_apex)
    print(num_basal)

    assert num_apex == num_basal, "Incompatible Graph Sizes"

    # community = g1.community_leading_eigenvector()
    # membership = community.membership
    


    # basal_respective_membership = []
    # trophic_level_indexer(g1)
    # for i in basals:
    #     selected_indices = g1.vs.select(name = g2.vs[i]["name"]).indices #recall that horzcat puts duplicates in the same module, so this shouldn't be a problem
    #     selected_indices = sorted(selected_indices, key = lambda x : g1.vs[x]["trophic_level"], reverse=True) 
    #     #Catches highest trophic level occurance, to avoid all new basals going to the module with the very first set of basals
    #     basal_respective_membership.append(membership[selected_indices[0]])
    
    # apex_respective_membership = []
    # for i in apexes:
    #     apex_respective_membership.append(membership[i])
    
    basal_zip = enumerate(basals)
    apex_zip = enumerate(apexes)

    #sorting only on degree, highest degrees get connected
    basal_zip = sorted(basal_zip, key=lambda x: (g2.degree(x[1]), x[0]))
    apex_zip = sorted(apex_zip, key=lambda x: (g1.degree(x[1]), x[0]))

    """
    basal_temp = []
    for i in range(len(apex_zip)):

        if i  > (len(basal_zip) - 1):
            basal_zip.append(None)
            
        if not basal_zip[i] == None:
            while apex_zip[i][1] > basal_zip[i][1]:
                basal_temp.append(basal_zip[i])
                basal_zip.pop(i)
            if apex_zip[i][1] < basal_zip[i][1]:
                basal_zip.insert(i, None)

    while len(basal_zip) > len(apex_zip):
            basal_temp.append(basal_zip[-1])
            basal_zip.pop(-1)
    

    basal_temp = sorted(basal_temp, key=lambda x: (x[1],g2.degree(x[0])))
    for temp in basal_temp:
        basal_zip[basal_zip.index(None)] = temp
    """

    
    
    origin_count = g1.vcount()
    g1.add_vertices(g2.vcount() - num_basal)

    new_edges = []

    #pop all basal indices from original list of species
    vertex_indices = list(range(g2.vcount()))

    for i in basals:
        vertex_indices.pop(vertex_indices.index(i))

    #vertex_indices will have a length of g2.vcount() - num_basals
    index_map = {old_index: origin_count + i for i, old_index in enumerate(vertex_indices)}

    for edge in g2.es:
        if edge.source in basals:

            index = next((i for i, x in enumerate(basal_zip) if x[1] == edge.source), None) #x[1] for the index, x[0] for the prior enumeration

            new_source = apex_zip[index][1]

            new_edges.append((new_source, index_map[edge.target]))
        else:
            new_edges.append((index_map[edge.source], index_map[edge.target]))

    g1.add_edges(new_edges)

    #Copy names over

    if "name" in g2.vs.attributes():
        for key, value in index_map.items():
            g1.vs[value]["name"] = g2.vs[key]["name"]

    return g1
        



    #for each basal species, find what 

    #essentially, want to group basals of g2 to the apexes of g1 by module, if possible




def longest_path_dag(graph):
    # Ensure the graph is a DAG
    if not graph.is_dag():
        raise ValueError("Graph is not a DAG")

    # Perform topological sorting
    topological_order = graph.topological_sorting()

    # Initialize distances to all vertices as -∞
    distances = [-float("inf")] * graph.vcount()
    distances[topological_order[0]] = 0  # Assume the first vertex in topological order is the source

    # Process vertices in topological order
    for vertex in topological_order:
        for neighbor in graph.successors(vertex):
            distances[neighbor] = max(distances[neighbor], distances[vertex] + 1)

    # Find the vertex with the maximum distance
    max_distance = max(distances)
    end_vertex = distances.index(max_distance)

    # Reconstruct the longest path
    path = [end_vertex]
    while distances[end_vertex] != 0:
        for predecessor in graph.predecessors(end_vertex):
            if distances[predecessor] == distances[end_vertex] - 1:
                path.append(predecessor)
                end_vertex = predecessor
                break
    

    return max_distance

# Function to find all cycles in the graph
def find_cycles(graph):
    cycles = []
    for vertex in range(graph.vcount()):
        stack = [(vertex, [vertex])]
        while stack:
            (current, path) = stack.pop()
            for neighbor in graph.successors(current):
                if neighbor in path:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                elif neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
    return cycles





def graph_to_nn(g, num_inputs, num_outputs, output_function = F.leaky_relu, device = 'cpu'):
    """
    Basis: num_inputs and num_outputs dictate the input/output mlp layer widths
    if one of them is zero, the network will connect directly to either the apex or basal layer
    """
    input_neurons = [Neuron() for _ in range(num_inputs)]  
    output_neurons = [Neuron(activation=output_function) for _ in range(num_outputs)]  # Two output neurons

    end_nodes = [v.index for v in g.vs if g.degree(v.index, mode="out") == 0 and g.degree(v.index, mode="in") > 0] #apex
    input_nodes = [v.index for v in g.vs if g.degree(v.index, mode="in") == 0 and g.degree(v.index, mode="out") > 0] #basal

       

    for i in range(0, g.vcount()):
        g.vs[i]["neuron"] = Neuron()

    all_neurons = input_neurons + list(g.vs["neuron"]) + output_neurons 

    for i in range(0, g.vcount()):
        if g.degree(i, mode = "in") == 0 and g.degree(i, mode = "out") > 0 and input_neurons:
            # If the neuron is a basal neuron, connect it to the input neurons
            g.vs[i]["neuron"].set_inputs(input_neurons, all_neurons)
        else:
            current_inputs = [g.vs[j]["neuron"] for j in g.neighbors(i, mode="in")]
            g.vs[i]["neuron"].set_inputs(current_inputs, all_neurons)

    current_inputs = [g.vs[j]["neuron"] for j in end_nodes]

    if output_neurons:
        for i in output_neurons:
            i.set_inputs(current_inputs, all_neurons)

    if not g.is_dag():
        all_cycles = find_cycles(g)
        order = []
    else:
        all_cycles = 0
        order = g.topological_sorting()
    largest_cycle = max(all_cycles, key=len) if all_cycles else 0

    #need a handle for when graph is not dag


    if largest_cycle == 0:
        largest_cycle = -1 * longest_path_dag(g) #if there are no cycles, we want the longest path in the graph

    elif largest_cycle: #in the event that it is a list of nodes, we want the number nodes
        largest_cycle = len(largest_cycle)
        

    if not input_neurons:
        #set num inputs to the number of basal species
        input_indices = input_nodes
    else:
        input_indices = list(range(num_inputs))
        order = input_indices + [x + len(input_indices) for x in order] #add input neurons to the order of the graph
    
    if not output_neurons:
        #set num outputs to the number of apex species
        output_indices = end_nodes
    else:
        output_indices =  [x + len(input_neurons + list(g.vs["neuron"])) for x in list(range(num_outputs))]
        order = order + output_indices #add output neurons to the order of the graph

    if not g.is_dag():
        order = None

    out = Network(all_neurons, largest_cycle = largest_cycle, order=order, input_indices = input_indices, output_indices = output_indices, device=device)
    
    print("Largest cycle length:", largest_cycle)

    return out
    

class Neuron(nn.Module):
    def __init__(self, activation=F.leaky_relu):
        super().__init__()
        self.activation = activation
        self.weight = None
        self.bias = nn.Parameter(torch.randn(1))

    def set_inputs(self, input_neurons: list, neuron_list: list):
        self.input_neurons = input_neurons
        self.input_indices = [neuron_list.index(n) for n in input_neurons]
        n_inputs = len(self.input_indices)
        self.weight = nn.Parameter(torch.randn(n_inputs))

        #nn.init.normal_(self.weight, mean = 0.0, std = 0.1)  # Initialize weights with Xavier uniform distribution
        temp_weight = self.weight.view(1, -1)  # Reshape to (1, n_inputs)
        nn.init.xavier_uniform_(temp_weight)
        self.weight.data = temp_weight.view(-1)  # Reshape back to 1D

    def forward_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        # Compute the output for a batch of inputs
        z = torch.matmul(inputs, self.weight) + self.bias
        return self.activation(z)

class Network(nn.Module):
    def __init__(self, neurons, input_indices, output_indices, order, largest_cycle=0, lr=0.001, device='cpu'):


        super().__init__()
        self.neurons = nn.ModuleList(neurons)
        self.input_neurons_idx = input_indices
        self.output_neurons_idx = output_indices
        self.num_neurons = len(neurons)
        self.order = order
        self.largest_cycle = largest_cycle
        self.lr = lr
        self.device = device

        #Get rid of neuron params
        for param in self.parameters():
            param.requires_grad = False

        self.weight_matrix = self.build_weight_matrix()
        self.weight_matrix.data = self.weight_matrix.data / torch.norm(self.weight_matrix.data, p=2)
        
        self.bias_vector = self.build_bias_vector()

        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr) #torch.optim.Adam(self.parameters(), lr=lr)
        self.scaler = GradScaler(self.device)


    
    def build_weight_matrix(self):
        W = torch.zeros(self.num_neurons, self.num_neurons, device=self.device)
        for i, neuron in enumerate(self.neurons):
            if i not in self.input_neurons_idx:
                for idx, j in enumerate(neuron.input_indices):
                    W[i, j] = neuron.weight[idx]
        return nn.Parameter(W)

    def build_bias_vector(self):
        return nn.Parameter(torch.tensor([n.bias.item() for n in self.neurons], device=self.device))

    def forward(self, x_batch, convergence=0.1, max_iters = 200, damping_factor = 0.8):
        batch_size = x_batch.size(0)
        all_outputs = torch.zeros(batch_size, self.num_neurons, dtype=x_batch.dtype, device=x_batch.device)
        # Set input neuron outputs
        all_outputs[:, self.input_neurons_idx] = x_batch[:, self.input_neurons_idx]

        # Batched forward pass for all neurons (for cyclic order)

        if self.largest_cycle > 0:

            for _ in range(max_iters):
                prev_outputs = all_outputs.clone()
                # Matrix multiplication for simultaneous firing
                all_inputs = torch.matmul(all_outputs, self.weight_matrix.T) + self.bias_vector
                all_outputs = F.leaky_relu(all_inputs)
                
                all_outputs *= damping_factor

                all_outputs[:, self.output_neurons_idx] = self.neurons[self.output_neurons_idx[-1]].activation(all_inputs[:, self.output_neurons_idx])
                all_outputs[:, self.input_neurons_idx] = x_batch[:, self.input_neurons_idx]  # Clamping basal inputs
                if torch.max(torch.abs(all_outputs - prev_outputs)) < convergence:
                    break
                if torch.max(torch.abs(all_outputs)) > 1e4:
                    logging.info("Divergence detected, stopping forward pass")
                    break


        # Batched forward pass for all neurons (for acyclic order)
        elif self.largest_cycle <= 0:
            for _ in range(abs(self.largest_cycle) + 1):
                prev_outputs = all_outputs.clone()
                # Matrix multiplication for simultaneous firing
                all_inputs = torch.matmul(all_outputs, self.weight_matrix.T) + self.bias_vector
                all_outputs = F.leaky_relu(all_inputs)
                
                all_outputs[:, self.output_neurons_idx] = self.neurons[self.output_neurons_idx[-1]].activation(all_inputs[:, self.output_neurons_idx])
                all_outputs[:, self.input_neurons_idx] = x_batch[:, self.input_neurons_idx]  # Clamping basal inputs


            #Firing by order:
            # for i in self.order:
            #     if i not in self.input_neurons_idx:
            #         inputs = all_outputs[:, self.neurons[i].input_indices]
            #         all_outputs[:, i] = self.neurons[i].forward_batch(inputs)

        # Return outputs for the output neurons
        return all_outputs[:, self.output_neurons_idx]

    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_step(self, x_batch, y_batch, loss_fn=nn.CrossEntropyLoss()):
        self.train()
        self.optimizer.zero_grad()

        # Mixed precision forward passs
        with autocast(self.device.type):
            y_pred = self(x_batch)
            loss = loss_fn(y_pred + 1e-4, y_batch)

        # Backward pass with GradScaler
        self.scaler.scale(loss).backward()

        if self.largest_cycle > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        #if self.order == None:
        #    with torch.no_grad():  # Ensure no gradients are computed during normalization
        #        self.weight_matrix.data = self.weight_matrix.data / torch.norm(self.weight_matrix.data, p=2)

        return loss.item()

        y_pred = self(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def to(self, device):
        super().to(device)
        return self

    def save(self, file_path):
        # Serialize the neurons and other network attributes
        torch.save({
            "state_dict": self.state_dict(),
            "neurons": self.neurons,  # Save the neurons
            "input_indices": self.input_neurons_idx,
            "output_indices": self.output_neurons_idx,
            "order": self.order,
            "largest_cycle": self.largest_cycle,
            "weight_matrix": self.weight_matrix.data.cpu(),
            "bias_vector": self.bias_vector.data.cpu()
        }, file_path)
        print(f"Network saved to {file_path}")

    @classmethod
    def load(cls, file_path, device='cpu'):
        # Load the checkpoint
        checkpoint = torch.load(file_path, map_location=device)
        
        # Reconstruct the network using the saved neurons and metadata
        net = cls(
            neurons=checkpoint["neurons"],  # Load the neurons
            input_indices=checkpoint["input_indices"],
            output_indices=checkpoint["output_indices"],
            order=checkpoint["order"],
            largest_cycle=checkpoint["largest_cycle"],
            

        )

        with torch.no_grad():
            net.weight_matrix.copy_(checkpoint["weight_matrix"].to(device))
            net.bias_vector.copy_(checkpoint["bias_vector"].to(device))

        # Load the state dictionary
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        net.to(device)
        print(f"Network loaded from {file_path}")
        return net

"""
Noteworthy adjustments for FNN/RNN fusion present in Antarctic:
- Added a regularization term in the loss +1-e8 to prevent bowing out into NAN numbers
- Added gradient clipping of norm = 1.0
- Set ALL activation functions to leaky_relu to avoid exploding gradient
        This happens with sigmoids bc of the output plateu
- Set num iterations to 200, convergence to 0.01, and damping factor to 0.9 
        damping factor applied to all outputs to reduce oscillations and recurrent explosions
        does this reduce the cyclical nature of the network?
- Storing and updating weights in a weight matrix (SYNCRONOUS updates)

Things to add:
- Restore previous state if NaN approached?
- When saving network, need to store the weights back into the neurons from the matrix
            or just store the matrix too.

"""


