from recommendation_graph import Graph
from data_utils import get_data, get_movie_company_data

# from tabulate import tabulate

# get_movie_company_data('movie-lens-25m')

RANDOM = 'random'
BASELINE = 'baseline'
KNN = 'KNN'
SVD = 'SVD'

def graph():
  producer_data, item_data, user_data, rating_data, production_data = get_data("movie-lens-25m")

  g = Graph(rating_data, production_data, k = 10)

  pop_df = g.find_initial_popularity()

  alpha = g.find_alpha(pop_df)
  group1, group2 = g.get_groups_by_popularity(pop_df, alpha)

  print('popular group size:', group1.shape)
  print('other group size:', group2.shape)

  print("getting recommendations")
  algo = SVD

  data = g.load_data(rating_data)
  # g.compare_models(data)

  recommendations = g.get_recs(data, item_data['itemId'], user_data, algo)
  g.modify_connections(zip(recommendations['itemId'],recommendations['userId']))

  v1 = g.find_group_visibility(group1)
  v2 = g.find_group_visibility(group2)

  print(algo)
  print('popular visibility: ', v1)
  print('not-popular visibility: ', v2)
  print('disparate visibility: ', v1-v2)

  # plot_data = g.group_data_for_plotting(pop_df)
  # print(plot_data)
  
  x_data = pop_df['provider']
  y_data = pop_df['numRatings']

  # y_data = grouped_data['numUsers']
  # x_data = grouped_data['numRatings']
  # g.plot_data(x_data, y_data, 'providers', 'num times rated', 'Popularity Distribution (ml-latest-small)', 
  #             'line', log = False)

# producer_data, item_data, user_data, rating_data, production_data = get_data("movie-lens-small")
graph()

# print(item_data.head(10))
# print(rating_data.head(10))
# print(production_data.head(10))