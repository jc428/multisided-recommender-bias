from recommendation_graph import Graph
from data_utils import get_data

# item_data, rating_data, producer_data, production_data = get_data("book-crossing")
producer_data, item_data, user_data, rating_data, production_data = get_data("movie-lens-small")

g = Graph(producer_data, item_data, user_data, rating_data, production_data)
popularity_data = g.find_initial_popularity(rating_data)
data = g.group_data_for_plotting(popularity_data)
x_data = data['num_items']
y_data = data['count']
# g.plot_data(x_data, y_data, 'num items', 'num of ratings', 'Initial Item Popularity (movie-lens small)')
alpha = g.find_alpha(popularity_data)
print(alpha)
group1, group2 = g.get_groups_by_popularity(popularity_data, alpha)
v1 = g.find_group_visibility(group1)
v2 = g.find_group_visibility(group2)
print(group1.head())
print(group2.head())
print('popular visibility: ', v1)
print('not-popular visibility: ', v2)

# g.find_individual_visibility('i')