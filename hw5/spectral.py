import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import seaborn as sns

sns.set()


class Spectral:
    """
    Performs spectral clustering from scratch. In spectral clustering, the affinity, and not the absolute location
     (i.e. k-means), determines what points fall under which cluster. Can only be applied to a graph of connected
      nodes.
    """

    def __init__(self, data, hidden_dim=2):
        """
        Constructs necessary parameter for Spectral.
        :param data: array of nodes to build edges from
        """
        self.data = data
        self.hidden_dim = hidden_dim

    def __call__(self):
        """
        Performs entire clustering algorithm, prints several matrices, and draws graph w/ corresponding clusters.
        :return: nothing
        """
        # construct a similarity graph
        G = self.construct_graph()
        # determine the Adjacency matrix W, Degree matrix D and the Laplacian matrix L
        L = self.get_matrices(G)
        # compute the eigenvectors of the matrix L
        e, v = self.compute_eigens(L)
        # using the second smallest eigenvector as input, train a k-means model and use it to classify the data
        labels = self.train_kmeans(e, v)
        # get color list
        colors = self.make_colors(labels)
        # draw
        self.draw_graph(G, node_color=colors)

    @staticmethod
    def make_colors(labels):
        """
        Utility function to map cluster labels to colors for plot.
        :param labels: label output from kmeans
        :return: mapped color list
        """
        colors = []
        for label in labels:
            if label == 0:
                colors.append('r')
            elif label == 1:
                colors.append('b')
            elif label == 2:
                colors.append('g')
        return colors

    @staticmethod
    def draw_graph(G, node_color='r'):
        """
        Utility function to draw connected node graph.
        :param node_color: list or single string for node colors
        :param G: graph to draw
        :return: nothing
        """
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color=node_color)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    def construct_graph(self):
        """
        Generates a graph and adds edges based on initialized data object.
        :return: graph
        """
        G = nx.Graph()
        G.add_edges_from(self.data)
        return G

    @staticmethod
    def get_matrices(G):
        """
        Computes adjacency matrix, degree matrix, and Laplacian matrix.
        :param G: graph
        :return: laplacian matrix
        """
        # adjacency matrix
        W = nx.adjacency_matrix(G)
        print('adjacency matrix:')
        print(W.todense())
        # construct the degree matrix
        D = np.diag(np.sum(np.array(W.todense()), axis=1))
        print('degree matrix:')
        print(D)
        # laplacian matrix, subtract adjacency matrix from degree matrix
        L = D - W
        print('laplacian matrix:')
        print(L)
        return L

    @staticmethod
    def compute_eigens(L):
        """
        Computes eigenvalues and eigenvectors of inputted Laplacian matrix.
        :param L: laplacian matrix
        :return: eigenvalues, eigenvectors
        """
        # compute the eigenvalues and eigenvectors of the laplacian
        e, v = np.linalg.eig(L)
        return e, v

    @staticmethod
    def train_kmeans(e, v):
        """
        Trains a k-means model using the second smallest eigenvector as input and uses it to classify the data.
        :param e: eigenvalues
        :param v: eigenvectors
        :return: cluster labels
        """
        # using the second smallest eigenvector, use K-means to classify the nodes based
        # off their corresponding values in the eigenvector
        i = np.where(e < 0.5)[0]
        U = np.array(v[:, i[1]])
        km = KMeans(init='k-means++', n_clusters=3)
        km.fit(U)
        print(km.labels_)
        return km.labels_
