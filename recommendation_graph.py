from collections import defaultdict
from surprise import Dataset, Reader
from surprise import BaselineOnly, SVD, KNNBaseline, KNNBasic, NormalPredictor
from surprise.model_selection import GridSearchCV, cross_validate

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
  def __init__(self, item_data, user_data, rating_data, production_data, k = 5):
    self._production_graph = defaultdict(set)
    self._recommendation_graph = defaultdict(set)
    self._graph = defaultdict(set)
    self.__users = user_data
    self.__numUsers = user_data.shape[0]
    self.__items = item_data
    self.__numItems = item_data.shape[0]
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

  def find_group_visibility(self, group, type):
    k = self._k
    userCount = 0

    numUsers = self.__numUsers
    numItems = self.__numItems
    
    if type == 'provider':
      for provider in group:
        userCount = userCount + self.find_individual_visibility(provider)
      v = userCount /(numItems*k)
    elif type == 'item':
      for item in group:
        userCount = userCount + self.find_individual_visibility(item)
      v = userCount/(numUsers * k)
    return v
  
  def find_disparate_visibility(self, g1, g2, type):
    if type == 'provider':
      s1 = .03
      s2 = .97
    elif type == 'item':
      s1 = .2
      s2 = .8

    v1 = self.find_group_visibility(g1, type) / s1
    v2 = self.find_group_visibility(g2, type) / s2
    return v1 - v2

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

  def find_initial_popularity(self, type):
    """ type: 'provider' or 'item' """
    popularity_df = self.find_all_visibilities(type).rename(columns={'visibility': 'numRatings'})
    popularity_df = popularity_df.sort_values(by=['numRatings'], ascending = False).reset_index(drop = True)
    popularity_df.reset_index(inplace=True)
    popularity_df = popularity_df.rename(columns={'index': type})

    return popularity_df

  def find_alpha(self, popularity_data, type):
    if type == 'provider':
      p = .97
    elif type == 'item':
      p = .9
    alpha = popularity_data['numRatings'].quantile(p)
    print(alpha)
    return alpha

  def get_groups_by_popularity(self, data, alpha):
    """ producers with visibility > alpha are popular """
    group1 = data['nodeLabel'][data['numRatings'] > alpha]
    group2 = data['nodeLabel'][data['numRatings'] <= alpha]
    return group1, group2
  
  def plot_data(self, x_data, y_data, x_label, y_label, title, plot_type, log = False):
    if (log):
      y_data = np.log2(y_data)
    if plot_type == 'line':
      plt.plot(x_data, y_data, 'o-')
    elif plot_type == 'bar':
      plt.bar(x_data,y_data)
    
    alpha1  = y_data.quantile(.70)
    alpha2 = y_data.quantile(.80)
    alpha3 = x_data.quantile(.1)
    alpha4 = x_data.quantile(.03)
    alpha5 = y_data.quantile(.98)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.axvline(x = alpha3, color = 'red')
    plt.show()

  def plot_predicted_vs_actual_score(self, predicted_scores, actual_scores):
    """ """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, constrained_layout = True, sharey = True)
    fig.suptitle('Predicted vs Actual Ratings')
    fig.text(0.5, 0.04, 'actual rating', ha='center')
    fig.text(0.04, 0.5, 'predicted rating', va='center', rotation='vertical')
    colors = ['tab:blue','tab:red','tab:green', 'tab:purple']
    axs = [ax1,ax2,ax3,ax4]

    for (algo, preds), color, ax in zip(predicted_scores.items(), colors, axs):
      print(algo, preds)
      merged_df = actual_scores.merge(preds, on = ['userId', 'itemId'])
      x = merged_df['rating_x']
      y = merged_df['rating_y']
      ax.scatter(x, y, s = 10, c = color)
      ax.set_title(algo)
      ax.grid(True)

    plt.show()

  def plot_visibility_vs_popularity(self, visibility, popularity) :

    merged = popularity.merge(visibility, how = 'left', on = 'nodeLabel')
    print(merged)
    vis_x = merged['provider']
    vis_y = visibility['visibility']
    plt.scatter(vis_x, vis_y, s = 10, c = 'red', label = 'visibility')

    pop_x = popularity['provider']
    pop_y = popularity['numRatings']
    plt.scatter(pop_x, pop_y, s = 10, c = 'blue', label = 'popularity')
    plt.xlabel('provider')
    plt.ylabel('num users reached')
    plt.title('Provider Visibility and Popularity (SVD)')
    plt.legend()
    plt.show()


  def select_model(self, rating_data, algorithm, param_grid):
    if algorithm == 'SVD':
      reader = Reader(rating_scale=(1,5))
      data = Dataset.load_from_df(rating_data[['userId', 'itemId', 'rating']], reader)
      gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
      gs.fit(data)
      print('best rmse: ', gs.best_score['rmse'])
      print(gs.best_params['rmse'])

      return gs.best_estimator['rmse']
    else:
      pass

  def load_data(self, rating_data):
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(rating_data[['userId', 'itemId', 'rating']], reader)
    
    return data
  
  def compare_models(self, data):
    algos = [NormalPredictor(), BaselineOnly(), SVD(), KNNBaseline()]
    print('cross validating random')
    for algo in algos:
      cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

  def fit(self, data, algorithm):
    """ rating_data: pandas dataframe with columns userId, itemId, rating"""

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

    algo.fit(trainset)

    return algo
  
  def get_all_preds(self, algo, item_ids, user_ids):
    preds = [(str(uid), str(iid), algo.predict(uid,iid).est)
      for uid in user_ids for iid in item_ids]

    df = pd.DataFrame(preds, columns=['userId', 'itemId', 'rating'])
    return df

  def get_top_k(self, algo, item_ids, user_ids):
    """Return the top-k recommendation for each user from a set of predictions.
    From https://surprise.readthedocs.io/en/stable/FAQ.html
    Args:
      predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.

    Returns:
      A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size k.
    """
    print('getting preds')

    top_k = defaultdict(list)
    k = self._k
  # reorder so run it for each user, then get recommendations (don't keep score for every single item)
    for uid in user_ids:
      # print('getting recs for {}'.format(uid))
      preds = [(iid, algo.predict(uid,iid).est) for iid in item_ids]
      preds.sort(key=lambda x: x[1], reverse=True)
      top_k[uid] = preds[:k]

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
    algo = self.fit(rating_data, algorithm)
    t1 = time.clock() 
    print("time elapsed to get predictions: ", t1-t0)
    top_k = self.get_top_k(algo, item_ids, user_ids)
    t2 = time.clock()
    print("time elapsed to get top_k: ", t2-t1)
    rec_edges = self.get_recommendation_edges(top_k)
    t3 = time.clock()
    print("time elapsed to get recommendation edges: ", t3-t2)
    print(rec_edges.head())
    return rec_edges
