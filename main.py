import pandas as pd
from queue import Queue
import time
# import networkx as nx
# import algorithmx
# from algorithmx.networkx import add_graph
# import matplotlib.pyplot as plt

class Graph():
    def __init__(self, vert, edges):
        self.vert = vert
        self.edges = edges
        self.adj_list = {}
        self.edge_list = {}
        self.vert_list = {}
        for e in edges:
            self.adj_list[e] = []
            if e.u.ID not in self.adj_list:
                self.adj_list[e.u.ID] = []
            self.adj_list[e.u.ID].append(e.v.ID)
        for e in edges:
            self.edge_list[e.ID] = e
        for v in vert:
            self.vert_list[v.ID] = v

class Edge():
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight
        self.ID = self.u.ID + self.v.ID

class Vertex():
    def __init__(self, distance, predecessor, ID):
        self.distance = distance
        self.predecessor = predecessor
        self.ID = ID

#creates verticies from the list of vertex names
def init_verticies(vertex_list):
    vertex_final = []
    for i in range(len(vertex_list)):
        vertex_final.append(Vertex(float('inf'), None, vertex_list[i]))
    return vertex_final

#creates edges from the *.csv file
def create_edges_new(source, dest, cost, verticies, vert_list):
    edge_list = []
    for i in range(len(cost)): # cost is good, need to find source and dest
        for j in range(len(vert_list)):
            if source[i] == verticies[j].ID:
                temp_source = verticies[j]
            if dest[i] == verticies[j].ID:
                temp_dest = verticies[j]
        edge_list.append(Edge(temp_source, temp_dest, cost[i]))
    return edge_list

def dijkstra(graph, start_node, end_node):
    unvisited_nodes = list(graph.vert) #open list
    shortest_path = {} #stores g value of nodes
    previous_nodes = {} # closed list

#this initializes the cost of the vertices to infinity and sets source cost to 0
    max_value = float('inf')
    for node in unvisited_nodes:
        shortest_path[node.ID] = max_value
    shortest_path[start_node] = 0
    #creates a counter for how many nodes were evaluted
    nodes_evaluated = 0
    while unvisited_nodes: #while there are still nodes left to visit
        #increments the counter
        nodes_evaluated += 1
        #this selects the node from the array with the smallest distance
        min_node = None
        for node in unvisited_nodes:
            if min_node == None:
                min_node = node
            elif shortest_path[node.ID] < shortest_path[min_node.ID]:
                min_node = node
        # terminate if the shortest path candidate is the target destination
        if min_node == get_source(graph, end_node):
            print('Dijkstra nodes evaluated = ' + str(nodes_evaluated))
            return previous_nodes, shortest_path
        if min_node.ID in graph.adj_list:#checking to make sure the node has out edges (three of them in this data set don't)
            #iterating through neighbors to update distances
            neighbors = graph.adj_list[min_node.ID]
            for neighbor in neighbors:
                tentative_value = shortest_path[min_node.ID] + graph.edge_list[min_node.ID+neighbor].weight
                #storing the node path and distances in the appropriate dictionaries
                if tentative_value < shortest_path[neighbor]:
                    shortest_path[neighbor] = tentative_value
                    previous_nodes[neighbor] = min_node.ID
        #remove the evaluated node from the pool
        unvisited_nodes.remove(min_node)
    #Displayed the counter of how many nodes were evaluated
    print('Dijkstra nodes evaluated = ' + str(nodes_evaluated))
    #return the shortest path and node path
    return previous_nodes, shortest_path


#returns a vertex given the vertex ID
def get_source(graph, node_id):
    node = graph.vert_list[node_id]
    return node

def bfs(graph, start_node, end_node):
    #initializes node list and assigns appropriate values
    total_nodes = list(graph.vert)
    max_value = float('inf')
    for node in total_nodes:
        node.distance = max_value
    s = get_source(graph, start_node)
    s.distance = 0
    #creates queue and puts source in queue
    pending_nodes = Queue()
    pending_nodes.put(s)

    while not pending_nodes.empty():
        #gets next node from queue
        node = pending_nodes.get()
        if node == get_source(graph, end_node):
            return graph
        if node.ID in graph.adj_list: #makes sure the node has out edges
            #assigns neighbors as the out edges of the current node
            neighbors = graph.adj_list[node.ID]
            #iterates through neighbors
            for neighbor in neighbors:
                next_node = graph.edge_list[node.ID+neighbor].v
                #updates neighbors distance value and predecessors
                if next_node.distance == max_value:
                    next_node.distance = node.distance + 1
                    next_node.predecessor = node
                    #calls the next node into the queue to be evaluated
                    pending_nodes.put(next_node)
    #returns the graph that the values can be derived from
    return (graph)

#prints the result of a BFS search. Not used in the other algorithms, but useful for testing and troubleshooting
def bfs_print(graph, start_node, end_node):
    path = []
    node = graph.vert_list[end_node]
    print("The fewest number of hops is " + str(node.distance))
    while node.ID != start_node:
        path.append(node.ID)
        node = node.predecessor
    source = graph.vert_list[start_node]
    path.append(source.ID)
    print(" -> ".join(reversed(path)))

#returns the number of hops from a start node to an end node(used as h(x)
def h_x_bfs_distance(graph, start_node, end_node):
    graph = bfs(graph, start_node, end_node)
    hops = get_source(graph, end_node).distance
    return hops

def a_star(graph, bfs_graph, start_node, end_node, min_cost):
    #initializes list of nodes and dictionaries that will record the path and score
    unvisited_nodes = list(graph.vert)
    shortest_path = {} #gscore
    fscore = {} #fscore
    previous_nodes = {}
    #initializes the values of the nodes
    max_value = float('inf')
    for node in unvisited_nodes:
        shortest_path[node.ID] = max_value
        fscore[node.ID] = max_value
        node.distance = max_value
    shortest_path[start_node] = 0
    fscore[start_node] = 0
    get_source(graph, start_node).distance = 0
    #a separate graph is used for bfs, because running bfs will change the values of the predecessors and the path
    get_source(bfs_graph, start_node).distance = 0
    #creates a counter for how many nodes were evaluated
    nodes_evaluated = 0
    while unvisited_nodes:
        #increments the counter
        nodes_evaluated += 1
        # print(random()) #just here to indicate that the process is running
        #selecting the node with the smallest fscore
        min_node = None
        for node in unvisited_nodes:
            if min_node == None:
                min_node = node
            elif fscore[node.ID] < fscore[min_node.ID]:
                min_node = node
        # terminate if the shortest path candidate is the target destination
        if min_node == get_source(graph, end_node):
            print('A* nodes evaluated = ' + str(nodes_evaluated))
            return previous_nodes, shortest_path

        elif min_node.ID in graph.adj_list: #making sure that there are outgoing edges from this node
            neighbors = graph.adj_list[min_node.ID] #list of 'v' nodes on outgoing edges from min_node

            for neighbor in neighbors: #iterating through all of the neighbors
                tentative_value = shortest_path[min_node.ID] + graph.edge_list[min_node.ID + neighbor].weight #this is g_x
                h_x = h_x_bfs_distance(bfs_graph, neighbor, end_node)*min_cost #calculating minimum hops from neighbor to destination
                f_x = tentative_value + h_x #adding minimum hops to current distance

                if tentative_value < shortest_path[neighbor]: #if the new value is shorter than the previously caclulated one
                    shortest_path[neighbor] = tentative_value #update the distance of the shortest path
                    fscore[neighbor] = f_x
                    previous_nodes[neighbor] = min_node.ID #add this to the path

        unvisited_nodes.remove(min_node) #remove it from the list of nodes to visit
    #prints the counter
    print('A* nodes evaluated = ' + str(nodes_evaluated))
    return previous_nodes, shortest_path


#print_result is used to display the distance and node path for dijkstra's and A* algorithm
def print_result(previous_nodes, shortest_path, source, dest):
    path = []
    node = dest
    while node != source:
        path.append(node)
        node = previous_nodes[node]
    path.append(source)
    print("The cheapest itinerary is $" + str(round(shortest_path[dest], 2)))
    print(" -> ".join(reversed(path)))

#reading the data form the *.csv files
flight_data = pd.read_csv (r'Flight_Data_Greater_Than_5.csv')
nodes = pd.read_csv(r'Nodes.csv')

#remove the two lines above and use these for troubleshooting, it is a simplified data set that makes testing quicker
# flight_data = pd.read_csv (r'C:\Users\patri\PycharmProjects\Dijkstra\Flight_Data_Simple.csv')
# nodes = pd.read_csv(r'C:\Users\patri\PycharmProjects\Dijkstra\Nodes_Simple.csv')

#sorting the data into separate arrays
source_list=flight_data['Source'].values
dest_list = flight_data['Dest'].values
cost_list = flight_data['Cost'].values
vert_list = nodes['Nodes'].values

#finding the cheapest flight in the dataset to use in the heuristic algorithm
min_cost = float('inf')
for i in range(len(cost_list)):
    if cost_list[i] < min_cost:
        min_cost = cost_list[i]



#converting the name strings from the data into Vertex types
verticies = init_verticies(vert_list)
#creating Edge types from teh data
edges = create_edges_new(source_list, dest_list, cost_list, verticies, vert_list)

#creating the graphs that will be fed into A*
standard_graph = Graph(verticies, edges)
bfs_graph = Graph(verticies, edges)

#assinging the source and destination nodes
start = "YAK"
end = "RUT"

starttimeA = time.time()
#running the A* algorithm
prev_nodes, path = a_star(standard_graph, bfs_graph, start, end, min_cost)

endtimeA = time.time()

#displaying results
print_result(prev_nodes, path, start, end)
print("Time Diff", endtimeA - starttimeA)
print("start", starttimeA)
print("end", endtimeA)



startd = time.time()
#running Dijkstra's algorithm
previous_nodes, shortest_path = dijkstra(standard_graph, start, end)
endd = time.time()


# displaying results
print_result(previous_nodes, shortest_path, start, end)
print(endd - startd)
print("Time Diff", endd - startd)
print("start", startd)
print("end", endd)
#running BFS algorithm
# throw = bfs(bfs_graph, start)
#displaying results
# bfs_print(bfs_graph, start, end)

#was playing around with visualization, it's difficult with the amount of nodes we have
# G = nx.DiGraph()
# G.add_nodes_from(node_list)
# # for i in range(len(source_list)):
# #     G.add_weighted_edges_from([(source_list[i], dest_list[i], cost_list[i])])
# G.add_weighted_edges_from([(1, 2, 0.4), (2, 3, 0.2), (3, 1, 0.3)])
# nx.draw(G, with_labels=True)
# plt.show()


