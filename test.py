from recommendation_graph import Graph, RECOMMENDATION
from data_utils import get_data, get_movie_company_data
# from tabulate import tabulate

# get_movie_company_data('movie-lens-25m')

def graph():
  producer_data, item_data, user_data, rating_data, production_data = get_data("movie-lens-100k")
  # producer_data, item_data, user_data, rating_data, production_data = get_data("movie-lens-25m")

  g = Graph(rating_data, production_data)

  pop_df = g.find_initial_popularity()
  print(pop_df.head())

  print('rating_df shape:\t{}'.format(rating_data.shape))
  # num_ratings_by_user = rating_data.reset_index().groupby('userId')['index'].count().reset_index(name = 'numRatings')
  # print('num ratings by user: ')
  # print(num_ratings_by_user.head())
  # grouped_data = num_ratings_by_user.groupby('numRatings')['userId'].count().reset_index(name = 'numUsers').sort_values(by=['numRatings'], ascending = False)
  # print(grouped_data)

  alpha = g.find_alpha(pop_df)
  # alpha = 400000
  group1, group2 = g.get_groups_by_popularity(pop_df, alpha)

  print("getting recommendations")
  recommendations = g.get_recs(rating_data, item_data['itemId'], user_data)
  g.modify_connections(zip(recommendations['itemId'],recommendations['userId']), RECOMMENDATION)

  v1 = g.find_group_visibility(group1)
  v2 = g.find_group_visibility(group2)

  print("baseline")
  print('popular visibility: ', v1)
  print('not-popular visibility: ', v2)

  plot_data = g.group_data_for_plotting(pop_df)
  # print(plot_data.head())
  # x_data = plot_data['numItems']
  # y_data = plot_data['numRatings']
  # y_data = rating_data[['userId','itemId']]

  # y_data = grouped_data['numUsers']
  # x_data = grouped_data['numRatings']
  # g.plot_data(y_data, x_data, 'num ratings','num users', 'User Rating Distribution (Book-Crossing)', 'bar')

def select_model():
  pass

producer_data, item_data, user_data, rating_data, production_data = get_data("movie-lens-100k")
# graph()
# print(item_data.head(10))
# print(rating_data.head(10))
# print(production_data.head(10))