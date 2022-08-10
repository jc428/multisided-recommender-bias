from recommendation_graph import Graph
from data_utils import get_data, get_movie_company_data
import sys

def graph():
  producer_data, item_data, user_data, rating_data, production_data = get_data("movie-lens-small")

  # create the multisided graph
  g = Graph(item_data['itemId'], user_data, rating_data, production_data, k = 10)

  p_pop_df = g.find_initial_popularity('provider')
  print(p_pop_df)

  alpha = g.find_alpha(p_pop_df, 'provider')
  pg1, pg2 = g.get_groups_by_popularity(p_pop_df, alpha)

  i_pop_df = g.find_initial_popularity('item')
  print(i_pop_df)
  alpha = g.find_alpha(i_pop_df, 'item')
  print(alpha)
  ig1, ig2 = g.get_groups_by_popularity(i_pop_df, alpha)

  # plot the number of ratings each item got
  g.plot_data(i_pop_df['item'], i_pop_df['numRatings'],
              'Item', 'Num times rated', 'Item Popularity Distribution', 'line')

  # for each provider plot the sum of the number of ratings each item got for 
  # every item produced by that provider
  g.plot_data(p_pop_df['provider'], p_pop_df['numRatings'],
              'Provider', 'Num times rated', 'Provider Popularity Distribution', 'line')

  print('popular group size:', pg1.shape)
  print('other group size:', pg2.shape)

  print('popular group size:', ig1.shape)
  print('other group size:', ig2.shape)

  data = g.load_data(rating_data)

  algorithms = ['random', 'baseline', 'SVD', 'KNN']

  preds_lst = {}

  original_stdout = sys.stdout

  # visibility results are written to output.txt
  with open('output.txt', 'w') as f:
    sys.stdout = f
    for algorithm in algorithms:
      # train the recommender for each algorithm
      algo = g.fit(data, algorithm)
      preds = g.get_all_preds(algo, item_data['itemId'], user_data)
      preds_lst[algorithm] = preds

      # get recommendation lists for each user and update the graph accordingly
      rec_edges = g.get_recs(data, item_data['itemId'], user_data, algorithm)
      g.modify_connections(zip(rec_edges['itemId'], rec_edges['userId']))

      # calculate group and disparate visibility for provider and item
      pv1 = g.find_group_visibility(pg1, 'provider')
      pv2 = g.find_group_visibility(pg2, 'provider')

      print('provider visibility')
      print('popular visibility: ', pv1)
      print('not-popular visibility: ', pv2)
      print('disparate visibility: ', g.find_disparate_visibility(pg1, pg2, 'provider'))

      iv1 = g.find_group_visibility(ig1, 'item')
      iv2 = g.find_group_visibility(ig2, 'item')

      print('item visibility')
      print('popular visibility: ', iv1)
      print('not-popular visibility: ', iv2)
      print('disparate visibility: ', g.find_disparate_visibility(ig1, ig2, 'item'))
    sys.stdout = original_stdout

  # for each algorithm plot the predicted value against the actual rating each item received in the dataset
  # g.plot_predicted_vs_actual_score(preds_lst, rating_data)

graph()

