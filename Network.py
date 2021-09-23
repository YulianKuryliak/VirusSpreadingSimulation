import igraph

# IDEAS:
# -> create class for nodes which have to contains information about the node
#   or find out how to use igraph for this


def complete_graph(size):
    return igraph.Graph.Full(size, directed=False, loops=False)

def ER_model(size, edges):
    return igraph.Graph.Erdos_Renyi(n = size, m = edges, loops = False, directed = False)

def WS_model(size, edges_per_node, reconnection_probability):
    return igraph.Graph.Watts_Strogatz(1, size, edges_per_node, reconnection_probability, loops = False, multiple = False)

def BA_model(size, edges_per_node):
    return igraph.Graph.Barabasi(size, edges_per_node, outpref = False, directed = False)

def set_population_info(graph):
    return graph #pass

def create_network(size, edges, model='complete', reconnection_probability=0):
    if(model == 'Barabasi-Albert' or model == 'BA'):
        graph = BA_model(size, edges)
    elif(model == 'Watts-Strogatz' or model == 'WS' or model == 'small world'):
        graph = WS_model(size, edges, reconnection_probability)
    elif(model == 'Erdos_Renyi' or model == 'ER' or model == 'random'):
        graph = ER_model(size, edges)
    else:
        graph = complete_graph(size)

    graph = set_population_info(graph)
    return graph

# graph = create_network(20, 2, 'BA')
# igraph.plot(graph, layout = graph.layout("circle"))
# print(graph.get_adjacency())