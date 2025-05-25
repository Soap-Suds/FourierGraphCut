import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from anytree import RenderTree
from src.lazy_branch_and_bound import LazyBranchAndBound

def test(data, k, initial_guess, showT = True, showG = True, updates = False, name = None):
    A = data[0]
    G = data[1]

    if showG:
        num_vertices = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # Calculate figure size based on the number of vertices and edges
        width = 8 + num_vertices * 0.1
        height = 6 + num_edges * 0.05

        plt.figure(figsize=(width, height))
        nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.title(name)
        plt.show()
        
        pass

    bb = LazyBranchAndBound(A, k, initial_guess)
    Optima = bb.search(updates)
    Max = float(round(Optima[0],2))
    Arg = Optima[1]

    count = 0
    for pre, fill, node in RenderTree(bb.root): #fill is just a filler variable because of how RenderTree works.
        count += 1
        if showT:
            if len(node.name[1]) < bb.n:
                bounds = bb._bound(node)  # Get the bounds [b1, b2, b3]
                print(f'{pre} {node.depth} {node.name[1]} {bounds}')
                #print(f'{pre} {node.depth} {node.name[1]} {float(round(node.bound,2))}')
            else: print(f'{pre} {node.depth} {node.name[1]} {float(round(bb.value(node),2))}')

    percentage_pruned = float(np.round(((1 - count * (bb.k - 1) / (bb.k**(bb.n - 1) - 1)) * 100), 2)) #How many nodes the tree pruned / didn't check.

    result = [Max,Arg,percentage_pruned]

    if showG:
        present(result, eff = True, max = True, arg = True)
    
    return result

def multitest(dataset, K, guesses, showG = True, showT = False, i=0, results=None):  
    if results is None:  
        results = [[], [], []]  # Initialize the results list

    if i >= len(K):  # Correctly compare integer i to length of K
        return results
    
    # Call test and store its result
    result = test(dataset[1][i], K[i], guesses[i], showG = showG, showT = showT, name = dataset[0])
    testmax = result[0]
    testarg = result[1]
    testeff = result[2]

    # Append the result (percentage pruned) to results
    results[0].append(testmax)
    results[1].append(testarg)
    results[2].append(testeff)
    
    # Continue recursion
    return multitest(dataset, K, guesses, showG, showT, i+1, results)  # Pass the results list down the recursion

def present(result, eff = True, max = False, arg =False):
    if eff:
        print("The percentage pruned for the graph is: " + str(result[2]))
    if max:
        print("The maxima found for the graph is: " + str(result[0]))
    if arg:
        print("The maximisers for the graph is: " + str(result[1]))
    return

def multipresent(results, eff = True, max = False, arg =False):
    if eff:
        print("The percentage pruned for each dataset is: " + str(results[2][:]))
    if max:
        print("The maxima found for each dataset are: " + str(results[0][:]))
    if arg:
        print("The maximisers for each dataset are: " + str(results[1][:]))
    return

def run_datasets(datasets, k_values, initial_guesses, showG=False):
    m = datasets['VS'][1].len() #Always run VS dataset
    cut_results = {k: {} for k in k_values}
    for k in k_values:
        for key in datasets.keys():
            cut_results[k][key] = multitest(datasets[key], [k for i in m], initial_guesses, showG=showG)
            print("Dataset: " + key)
    return cut_results

def statistics(results):
    mean = float(round(np.mean(results[2]), 2))
    median = float(round(np.median(results[2]), 2))
    std = float(round(np.std(results[2]), 2))
    return [mean, median, std]

def create_output_dict(cut_results):
    k_values = cut_results.keys()
    keys = cut_results[2].keys() #Always compute one cut problem, k = 2.
    stats = {k: {} for k in k_values}
    for k in k_values:
        for key in keys:
            stats[k][key] = statistics(cut_results[k][key])
    return stats

def convert_to_dataframe(stats_dict):
    dataframes = {k: pd.DataFrame(stats_dict[k]) for k in stats_dict}
    return dataframes

def save_dataframes_to_csv(dataframes, base_trial, directory='Data'):
    # Create the base string 'trial'
    trial = f'{base_trial}_{get_next_available_index(directory, base_trial)}'
    directory = f'{directory}/{trial}'
    os.makedirs(directory, exist_ok=True)

    # Save the dataframes to CSV files
    for name, df in dataframes.items():
        df.to_csv(f'{directory}/{name}.csv')

def get_next_available_index(directory, base_trial):
    i = 0
    while os.path.exists(f'{directory}/{base_trial}_{i}'):
        i += 1
    return i