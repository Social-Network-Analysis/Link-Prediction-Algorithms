import networkx as nx
import numpy
import numpy as np
from others import normalize,recall_funct
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import time
import random

var_dict = {}
TW = {}
nodes_per_label = {}

def data_to_adj_review(dataset):
    h=nx.read_gml(dataset,label='id')
    adj_mat_s= nx.adjacency_matrix(h)
    n=adj_mat_s.shape[0]
    print (n)
    adj_mat_d=adj_mat_s.todense()
    x=h.degree()
    degree_matrix = np.identity(n)
    degree_matrix_2 = np.identity(n)
    for i in range(1,n):
        # print (i)
        degree_matrix[i-1,i-1] = x[i]
        if x[i] != 0 :
            degree_matrix_2[i-1,i-1] = 1/(x[i]**0.5)
    adj_orig = adj_mat_d
    adj_mat_i=adj_mat_d+np.identity(n)
    input_mat = np.matmul(adj_mat_i,degree_matrix)
    return input_mat, degree_matrix_2, adj_orig

def recall_funct (test_preds, test_edge_labels):
    test_pred_copy = np.copy(test_preds)
    a = np.mean(test_pred_copy)
    # print (a)
    for i in range (len(test_pred_copy)):
        if test_pred_copy[i]< a :
            test_pred_copy[i] = 0
        else :
            test_pred_copy[i] = 1
    recall_value = recall_score(test_edge_labels, test_pred_copy)
    acc_score_value = accuracy_score(test_edge_labels,test_pred_copy)
    return (recall_value,acc_score_value)

def normalize (n):
    max = 0
    for i in range(len(n)):
        for j in range(len(n)) :
            if max < n[i][j] : max = n[i][j]
    for i in n:
        if max > 0 :
            for j in range(len(i)):
                i[j] = i[j] / max
    return n


def max_comm_label (node):
    global var_dict
    G = var_dict['graph']
    all_labels = set()
    for node_neighbour in G.neighbors(node):
        all_labels.add(var_dict[node_neighbour])
    prob_actual = 1
    label_actual = var_dict[node]
    for label in all_labels:
        prob_new = 1
        for node_chk in G.neighbors(node):
            if var_dict[node_chk] == label :
                chk = 0
                if G.has_edge(node,node_chk) :
                    chk = G[node][node_chk]['weight']
                if var_dict['influence'][node][node_chk] == 1 :
                    prob_new = prob_new * (1 - chk)
        if prob_new < prob_actual :
            prob_actual = prob_new
            label_actual = label
            var_dict[node] = label
    return label_actual

def detachability (label) :
    global var_dict
    G = var_dict['graph']
    internal = 0
    external = 0
    DZ = 0
    for node in G :
        if var_dict[node] == label :
            for node_neighbour in G.neighbors(node) :
                if var_dict[node_neighbour] == label :
                    internal = internal + G[node][node_neighbour]['weight']
                else :
                    external = external + G[node][node_neighbour]['weight']
    if internal + external != 0 :
        DZ = internal / (internal + external)
    return DZ

def isolability_measure_single_label (label) :
    global var_dict
    G = var_dict['graph']
    isolability = 0
    internal = 0
    external = 0
    for node in G :
        if var_dict[node] == label :
            for node_neighbour in G.neighbors(node) :
                if (var_dict[node] == var_dict[node_neighbour]) :
                    internal = internal + G[node][node_neighbour]['weight']
                else :
                    external = external + G[node][node_neighbour]['weight']
    if external != 0 : isolability = internal/external
    return isolability

def external_density_eval () :
    global  var_dict
    G = var_dict['graph']
    numerator = 0
    n = len(G)
    denominator = n * (n-1)
    total_labels = set()
    for node in G:
        total_labels.add(var_dict[node])
    for label in total_labels :
        nodes_per_label = 0
        for node in G :
            if var_dict[node] == label : nodes_per_label += 1
        denominator -= (nodes_per_label * (nodes_per_label - 1))
    for (node1,node2) in G.edges() :
        if var_dict[node1] != var_dict[node2] and node1 != node2 : numerator += 1
    if denominator != 0 : return numerator/denominator
    else : return 0

def coverage_eval () :
    global var_dict
    G = var_dict['graph']
    numerator = 0
    denominator = len(G.edges)
    for (node1,node2) in G.edges() :
        if var_dict[node1] == var_dict[node2] and node1 != node2 : numerator += 1
    if denominator != 0:
        return numerator / denominator
    else:
        return 0

def modularity_eval () :
    global var_dict
    G = var_dict['graph']
    total_edges = len(G.edges)
    total_labels = set()
    for node in G:
        total_labels.add(var_dict[node])
    modularity = 0
    internal_final = 0
    external_final = 0
    for label in total_labels :
        internal = 0
        external = 0
        for node in G :
            if var_dict[node] == label :
                for node_neighbours in G.neighbors(node) :
                    if var_dict[node_neighbours] == label : internal += 1
                    else : external += 1
        internal_final += internal
        external_final += external
    internal_final = internal_final/2
    external_final = external_final/2
    modularity_final  = (internal_final/total_edges) - ((external_final*external_final)/(total_edges*total_edges))
    if (internal_final + external_final) == total_edges:
        print("modularity edge check correct")
    return modularity_final

def clustering_ss_eval(graph,tao,theta,file_name):
    print("inside clustering_ss_eval")
    global var_dict
    global TW
    global nodes_per_label
    adj = nx.adjacency_matrix(graph).todense()
    G = graph.copy()
    for (u, v) in G.edges():
        random_value = random.uniform(0, 1)
        G.edges[u, v]['weight'] = random_value
    var_dict['graph'] = G
    prev = numpy.zeros((len(adj), len(adj)))
    print("No. of edges - "+str(len(G.edges())))
    i = 1
    A = numpy.zeros((len(adj), len(adj)))
    var_dict['influence'] = A
    for i in range(len(adj)):
        for j in range(len(adj)):
            A[i][j] = -1
    for node in G:
        var_dict[node] = node
        for node_neighbour in G.neighbors(node):
            if G.edges[node,node_neighbour]['weight'] > random.uniform(0,1) : A[node][node_neighbour] = 1
            else : A[node][node_neighbour] = 0
    i = 0
    while i <= tao :
        print("for i = "+str(i))
        for node in G:
            old_label = var_dict[node]
            new_label = max_comm_label(node)
            var_dict[node] = new_label
        i = i + 1
        total_labels = set()
        for node in G:
            total_labels.add(var_dict[node])
        print("number of labels left "+str(len(total_labels)))
    all_labels = set()
    for node in G:
        all_labels.add(var_dict[node])
    for label in all_labels:
        DZ1 = detachability(label)
        if DZ1 < theta :
            just_neighbour = set()
            outer = set()
            TW = numpy.zeros(len(adj))
            for node in G:
                if var_dict[node] == label :
                    for node_neighbour in G.neighbors(node):
                        just_neighbour.add(node_neighbour)
                        if var_dict[node_neighbour] != label :
                            TW[node_neighbour] += 1
                else :
                    outer.add(node)
            NE = just_neighbour.intersection(outer)
            c_max = 0
            for node_inter in NE :
                if TW[node_inter] > c_max :
                    c_max = TW[node_inter]
            NS = set()
            for node_inter in NE:
                if TW[node_inter] == c_max :
                    NS.add(node_inter)

            CS_label = set()
            for node in NS :
                CS_label.add(var_dict[node])
            MID = -99999
            new_label = label
            for label_other in CS_label :
                factor2 = detachability(label_other)
                to_be_changed = set()
                for node in G :
                    if var_dict[node] == label_other :
                        to_be_changed.add(node)
                        var_dict[node] = label
                factor1 = detachability(label)
                for node in to_be_changed :
                    var_dict[node] = label_other
                TID = factor1 - factor2
                if TID > MID :
                    MID = TID
                    new_label = label_other
            for node in G :
                if var_dict[node] == label :
                    var_dict[node] = new_label

    total_labels = set()
    for node in G :
        total_labels.add(var_dict[node])
    total_isolability = 0
    for label in total_labels :
        total_isolability = total_isolability + isolability_measure_single_label(label)
    average_isolability = total_isolability/len(total_labels)
    external_density = external_density_eval()
    coverage = coverage_eval()
    modularity = modularity_eval()
    print("number of labels left finally " + str(len(total_labels)))
    return [len(total_labels),average_isolability,external_density,coverage,modularity]

def clustering_ss(graph):
    print("inside clustering_ss")
    global var_dict
    global TW
    global nodes_per_label
    tao = var_dict['tao']
    theta = var_dict['theta']
    adj = nx.adjacency_matrix(graph).todense()
    G = var_dict['graph'].copy()
    prev = numpy.zeros((len(adj), len(adj)))
    print("No. of edges - "+str(len(G.edges())))
    i = 1
    A = []
    A = numpy.zeros((len(adj), len(adj)))
    var_dict['influence'] = A
    for i in range(len(adj)):
        for j in range(len(adj)):
            A[i][j] = -1
    for node in G:
        var_dict[node] = node
        for node_neighbour in G.neighbors(node):
            if G.edges[node,node_neighbour]['weight'] > random.uniform(0,1) : A[node][node_neighbour] = 1
            else : A[node][node_neighbour] = 0
    i = 0
    while i <= tao :
        print("for i = "+str(i))
        for node in G:
            old_label = var_dict[node]
            new_label = max_comm_label(node)
            var_dict[node] = new_label
        i = i + 1
        total_labels = set()
        for node in G:
            total_labels.add(var_dict[node])
        print("number of labels left "+str(len(total_labels)))
    all_labels = set()
    for node in G:
        all_labels.add(var_dict[node])
    for label in all_labels:
        DZ1 = detachability(label)
        if DZ1 < theta :
            just_neighbour = set()
            outer = set()
            TW = numpy.zeros(len(adj))
            for node in G:
                if var_dict[node] == label :
                    for node_neighbour in G.neighbors(node):
                        just_neighbour.add(node_neighbour)
                        if var_dict[node_neighbour] != label :
                            TW[node_neighbour] += 1
                else :
                    outer.add(node)
            NE = just_neighbour.intersection(outer)
            c_max = 0
            for node_inter in NE :
                if TW[node_inter] > c_max :
                    c_max = TW[node_inter]
            NS = set()
            for node_inter in NE:
                if TW[node_inter] == c_max :
                    NS.add(node_inter)
            CS_label = set()
            for node in NS :
                CS_label.add(var_dict[node])
            MID = -99999
            new_label = label
            for label_other in CS_label :
                factor2 = detachability(label_other)
                to_be_changed = set()
                for node in G :
                    if var_dict[node] == label_other :
                        to_be_changed.add(node)
                        var_dict[node] = label
                factor1 = detachability(label)
                for node in to_be_changed :
                    var_dict[node] = label_other
                TID = factor1 - factor2
                if TID > MID :
                    MID = TID
                    new_label = label_other
            for node in G :
                if var_dict[node] == label :
                    var_dict[node] = new_label

    total_labels = set()
    for node in G :
        total_labels.add(var_dict[node])
    print("number of labels left finally " + str(len(total_labels)))
    for label in total_labels :
        count = 0
        nodes_per_label[label] = count
        for node in G :
            if var_dict[node] == label :
                count += 1
        nodes_per_label[label] = count
    cluster_matrix = numpy.zeros((len(adj), len(adj)))
    no_of_nodes = G.number_of_nodes()
    for i in range(len(adj)):
        for j in range(len(adj)) :
            if var_dict[i] == var_dict[j] :
                cluster_matrix[i][j] = int(nodes_per_label[var_dict[i]]) / no_of_nodes
            else :
                cluster_matrix[i][j] = int(nodes_per_label[var_dict[i]]) / no_of_nodes*-1
    var_dict['cluster_matrix_check'] = cluster_matrix
    return cluster_matrix

def clustering_main (adj) :
    G = nx.Graph(adj)
    var_dict['graph'] = G
    for (u, v) in G.edges():
        value = random.uniform(0, 1)
        G.edges[u, v]['weight'] = value
    cluster_matrix = clustering_ss(G)
    var_dict['cluster_matrix'] = cluster_matrix
    similarity_matrix = numpy.zeros((len(adj), len(adj)))
    overall_similarity_matrix = numpy.zeros((len(adj), len(adj)))
    print("making similarity matrix")
    for node1 in G :
        for node2 in G :
            similarity_matrix[node1][node2] = 1
            common_neighbour_factor = 1
            for node_neighbour in G.neighbors(node1) :
                if G.has_edge(node2,node_neighbour) :
                    common_neighbour_factor = common_neighbour_factor * (1 - G.edges[node2, node_neighbour]['weight'])
            neighbour_factor = 0
            if G.has_edge(node1, node2): neighbour_factor = G.edges[node1, node2]['weight']
            similarity_matrix[node1][node2] = 1 - common_neighbour_factor + neighbour_factor
    similarity_matrix = normalize(similarity_matrix)
    var_dict['similarity_matrix'] = similarity_matrix
    print("making overall similarity matrix")
    for i in range(len(adj)) :
        for j in range(len(adj)) :
            overall_similarity_matrix[i][j] = similarity_matrix[i][j]*cluster_matrix[i][j]
    var_dict['overall_similarity_matrix'] = overall_similarity_matrix
    link_pred = numpy.zeros((len(adj), len(adj)))
    print("making link prediction matrix")
    for node1 in G :
        for node2 in G :
            node_neighbour_common = nx.common_neighbors(G,node1,node2)
            for common_node in node_neighbour_common:
                link_pred[node1][node2] += overall_similarity_matrix[node1][common_node] + overall_similarity_matrix[common_node][node2]
    print("returning link prediction matrix")
    return link_pred

def clustering_recall_correct (graph_original,file_name,tao,theta,ratio):
    print("running correct clp_id")
    print("old number of edges - " + str(len(graph_original.edges)) + " for ratio - " + str(ratio))
    # making original graph adjacency matrix
    adj_original = nx.adjacency_matrix(graph_original).todense()
    starttime = time.time()
    # finding edges and nodes of original graph
    edges = np.array(list(graph_original.edges))
    nodes = list(range(len(adj_original)))
    np.random.shuffle(edges)
    edges_original = edges
    edges_train = np.array(edges_original, copy=True)
    np.random.shuffle(edges_train)
    # finding training set of edges according to ratio
    np.random.shuffle(edges_train)
    edges_train = random.sample(list(edges_train), int(ratio * (len(edges_train))))
    graph_train = nx.Graph()
    # making graph based on the training edges
    graph_train.add_nodes_from(nodes)
    graph_train.add_edges_from(edges_train)
    adj_train = nx.adjacency_matrix(graph_train).todense()
    print("new number of edges - " + str(len(graph_train.edges)) + " for ratio - " + str(ratio))
    # sending training graph for matrix prediction
    var_dict['tao'] = tao
    var_dict['theta'] = theta
    pred = clustering_main(adj_train)
    pred = normalize(pred)
    endtime = time.time()
    print('{} before precision'.format(endtime - starttime))
    # making test graph by removing train edges from original
    graph_test = nx.Graph()
    graph_test.add_nodes_from(nodes)
    graph_test.add_edges_from(edges_original)
    graph_test.remove_edges_from(edges_train)
    adj_test = np.zeros(shape=(len(adj_original), len(adj_original)))
    # making adcancecy test from testing graph
    adj_test = nx.adjacency_matrix(graph_test).todense()
    starttime = endtime
    # making new arrays to pass to function
    array_true = []
    array_pred = []
    for i in range(len(adj_original)):
        for j in range(len(adj_original)):
            if not graph_original.has_edge(i, j) :
                array_true.append(0)
                array_pred.append(pred[i][j])
            if graph_test.has_edge(i, j):
                array_true.append(1)
                array_pred.append(pred[i][j])
    # flattening adjacency matrices
    pred = pred.flatten()
    adj_original = np.array(adj_original).flatten()
    adj_test = np.array(adj_test).flatten()
    pred = array_pred
    adj_test = array_true
    # calculating auc and average preceision
    auc = roc_auc_score(adj_test, pred)
    prec = average_precision_score(adj_test, pred)
    # return precision recall pairs for particular thresholds
    prec_per, recall_per, threshold_per = precision_recall_curve(adj_test, pred)
    prec_per = prec_per[::-1]
    recall_per = recall_per[::-1]
    endtime = time.time()
    print('{} after precision before recall'.format(endtime - starttime))
    starttime = endtime
    recall = recall_funct(pred, adj_test)
    endtime = time.time()
    print('{} after recall'.format(endtime - starttime))
    file_inside = open('./result_range_review/' + file_name + ".txt", 'a')
    result = "\nclp_id"+"\nFor tao - "+str(tao)+" and for theta - "+str(theta)+"\nRatio :"+str(ratio)
    file_inside.write(result)
    file_inside.close()
    recall = recall_funct(pred, adj_test)[0]
    acc_score = recall_funct(pred, adj_test)[1]
    return (np.trapz(prec_per, x=recall_per), recall, auc, prec, acc_score)