from collections import defaultdict
from lib2to3.pytree import Base
from surprise import Dataset, Reader, accuracy
from surprise import BaselineOnly, SVD, KNNBaseline, NormalPredictor
from surprise.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class NodeTypeError(Exception):
  pass

PRODUCTION = 'production'
RECOMMENDATION = 'recommendation'

# random walk user -> item -> user -> item
# collapse into items only graph

#k is num of recommendations
class Graph(object):
  def __init__(self, rating_data, production_data, k = 5):
    self._production_graph = defaultdict(set)
    self._recommendation_graph = defaultdict(set)
    self._graph = defaultdict(set)
    # self._producer_nodes = []
    # self._item_nodes = []
    # self._user_nodes = []
    self._group1 = set()
    self._group2 = set()
    self._k = k
    self.add_connections(zip(production_data['producerId'],production_data['itemId']), PRODUCTION)
    # recommendations = self.get_initial_recs(rating_data, item_data['itemId'], user_data)
    self.add_connections(zip(rating_data['itemId'], rating_data['userId']), RECOMMENDATION)
    # self.add_connections(zip(recommendations['itemId'],recommendations['userId']), RECOMMENDATION)
    
  def add_connections(self, connections, edge_type):
    """Add edges to subgraph of graph corresponding to the edge type

    Args:
      connections(zip object): zip object with the first iterable element
        containing ids for left nodes and the second iterable element containing
        ids for right nodes

      edge_type(str): must be either PRODUCTION or RECOMMENDATION

    Returns:
      None
    """

    if edge_type == PRODUCTION:
      l_prefix = 'p'
      r_prefix = 'i'
      subgraph = self._production_graph
    elif edge_type == RECOMMENDATION:
      l_prefix = 'i'
      r_prefix = 'u'
      subgraph = self._recommendation_graph
    else:
      raise NodeTypeError('node must be provider or item')
    for left, right in connections:
      left = l_prefix + left
      right = r_prefix + right
      self.add_edge(left, right, subgraph)
  
  def modify_connections(self, new_connections):
    self._recommendation_graph = defaultdict(set)
    self.add_connections(new_connections, RECOMMENDATION)

  def add_edge(self, node1, node2, graph):
    graph[node1].add(node2)

  def get_out_degree(self, node):
    node_type = node[0]
    if node_type == 'i':
      subgraph = self._recommendation_graph
    elif node_type == 'p':
      subgraph = self._production_graph

    return len(subgraph[node])

  def get_node_info(self, node):
    return self._graph[node]

  def find_individual_visibility(self, node):
    """ node must be provider or item nodes """
    node_type = node[0]
    if node_type == 'i':
      return self.get_out_degree(node)
    elif node_type == 'p':
      v = 0
      for item in self._production_graph[node]:
        v = v + self.find_individual_visibility(item)
      return v
    else:
      raise NodeTypeError('node must be provider or item')

  def find_group_visibility(self, group):
    sum = 0
    for provider in group:
      sum = sum + self.find_individual_visibility(provider)
    v = sum/(len(group)*self._k)
    return v
  
  def find_disparate_visibility(self, g1, g2):
    v1 = self.find_group_visibility(g1)
    v2 = self.find_disparate_visibility(g2)
    return v1/len(g1) - v2/len(g2)

  def find_all_visibilities(self, node_type):
    if node_type == 'item':
      subgraph = self._recommendation_graph
    elif node_type == 'provider':
      subgraph = self._production_graph
    else:
      raise NodeTypeError('node type must be item or provider')
    arr = [[node,self.find_individual_visibility(node)] for node,value in subgraph.items()]
    df = pd.DataFrame(arr, columns = ['nodeLabel', 'visibility'])
    return df

  def find_initial_popularity(self):
    popularity_df = self.find_all_visibilities('provider').rename(columns={'visibility': 'numRatings'})

    return popularity_df

  def group_data_for_plotting(self, data):
    grouped_data = data.groupby('numRatings')['nodeLabel'].count().reset_index(name = 'numItems').sort_values(by=['numRatings'], ascending = False)
    # grouped_data['numItems'] = grouped_data['numItems'].cumsum()
    # print(sorted_data)
    return grouped_data

  def find_alpha(self, popularity_data):
    alpha = popularity_data['numRatings'].quantile(.8)
    print(alpha)
    return alpha
  
  def plot_data(self, x_data, y_data, x_label, y_label, title, plot_type, log = False):
    if (log):
      x_data = np.log2(x_data)

    if plot_type == 'line':
      plt.plot(x_data, y_data)
    elif plot_type == 'bar':
      x_data = x_data.clip(0,50)
      plt.bar(x_data,y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()



  def get_groups_by_popularity(self, data, alpha):
    """ producers with visibility > alpha are popular """
    group1 = data['nodeLabel'][data['numRatings'] > alpha]
    group2 = data['nodeLabel'][data['numRatings'] <= alpha]
    return group1, group2

  def get_predictions(self, rating_data, item_ids, user_ids, algorithm):
    """ rating_data: pandas dataframe with columns userId, itemId, rating"""
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(rating_data[['userId', 'itemId', 'rating']], reader)
    # trainset, testset = train_test_split(data, test_size=.25)
    trainset = data.build_full_trainset()

    if algorithm == 'baseline':
      algo = BaselineOnly()
    elif algorithm == 'random':
      algo = NormalPredictor()
    elif algorithm == 'SVD':
      algo = SVD()
    elif algorithm == 'KNN':
      algo = KNNBaseline()
    else:
      raise ValueError("algorithm must be one of 'baseline', 'random', 'SVD', or 'KNN'")

    print('fitting')
    algo.fit(trainset)
    
    print('training')

    preds = [algo.predict(uid,iid) for iid in item_ids
                    for uid in user_ids]
    
    return preds

  def get_top_k(self, predictions):
    """Return the top-k recommendation for each user from a set of predictions.
    From https://surprise.readthedocs.io/en/stable/FAQ.html
    Args:
      predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.

    Returns:
      A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size k.
    """
    top_k = defaultdict(list)
    k = self._k
    for uid, iid, true_r, est, _ in predictions:
      top_k[uid].append((iid, est))

    print('getting top k...')
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_k.items():
      user_ratings.sort(key=lambda x: x[1], reverse=True)
      top_k[uid] = user_ratings[:k]

    return top_k

  def get_recommendation_edges(self, top_k):
    """ Return all edges (item, user) where the item is recommended to user

    Args:
      top_k (dict): A dict where keys are user (raw) ids and values are lists of
        tuples: [(raw item id, rating estimation), ...] of size k.

    Returns: 
      A dataframe with columns ['itemId','userId].
    
    """
    print('getting recommendation edges...')
    lst = [[str(iid), str(uid)] for uid, recs in top_k.items() for iid, est in recs ]
    df = pd.DataFrame(lst, columns=['itemId','userId'])
    return df

  def get_recs(self, rating_data, item_ids, user_ids, algorithm = 'baseline'):
    t0 = time.clock()
    preds = self.get_predictions(rating_data, item_ids, user_ids, algorithm)
    t1 = time.clock() 
    print("time elapsed to get predictions: ", t1-t0)
    top_k = self.get_top_k(preds)
    t2 = time.clock()
    print("time elapsed to get top_k: ", t2-t1)
    rec_edges = self.get_recommendation_edges(top_k)
    t3 = time.clock()
    print("time elapsed to get recommendation edges: ", t3-t2)
    print(rec_edges.head())
    return rec_edges
