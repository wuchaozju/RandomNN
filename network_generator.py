import networkx as nx
import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

def network(n_nodes, n_edges):
	g = nx.barabasi_albert_graph(n_nodes, n_edges, seed=None)
	h = g.to_directed()

	#print(list(g.nodes))
	#nx.draw(g, pos=nx.circular_layout(g))
	#plt.savefig('network.png')

	return h

