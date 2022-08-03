import os, requests, json, csv

import pandas as pd
import unicodedata

DATA_DIR = './datasets/'
API_KEY = '0ab37300617849845e65d552a07cdeee'


def read_csv(dir_name, data_name, csv_params):
  """Load csv into pandas dataframe.
  
  Parameters
    zipped_dir_name: str
      The directory within the zip.
    data_name: str
      The name of the data file to be loaded from the directory.
    csv_params: str
      Parameters for loading csv into dataframe
  
  Returns
    data: DataFrame
  """
  filepath = os.path.join(DATA_DIR, dir_name, data_name)
  return pd.read_csv(filepath, **csv_params)

def get_tmdb_data(tmdb_id):
  """ """
  query = ('https://api.themoviedb.org/3/movie/' + tmdb_id + '?api_key='
    + API_KEY + '&language=en-US')
  response = requests.get(query)
  if response.status_code == 200:
    #successful query
    return json.dumps(response.json())
  else: 
    return 'error'
  
def get_companies(tmdb_id, n, companies):
  """ Get the names of production companies for a given movie. 
  
  Parameters
    tmdb_id: str
      id of movie in tmdb
  
  Returns
    company_lst: str list
      list of production companies
    company_ids: 
  """
  #tmdb data json in string form
  
  text = get_tmdb_data(tmdb_id)
  if text != 'error':
    info_obj = json.loads(text)
    try:
      company_lst = info_obj['production_companies']
    except:
      company_lst = []
    # print(company_lst)
    company_ids = []
    company_lst = list(map(lambda obj: obj['name'], company_lst))
    for company in company_lst:
      if company in companies:
        idx = companies[company]
        company_ids.append(idx)
      else:
        n = n + 1
        idx = n
        company_ids.append(idx)
        companies[company] = idx
        write_company_to_file('movie-lens-25m', idx, company)
    return company_lst, company_ids, n, companies
  return None, None, n, companies
  
def write_items_companies_to_file(data_name, company_ids, movie_id):
  if data_name == 'movie-lens-small':
    dir_name = 'ml-latest-small-company'
    filepath = os.path.join(DATA_DIR, dir_name, 'movies-companies.csv')
  elif data_name == 'movie-lens-25m':
    dir_name = 'ml-25m-company'
    filepath = os.path.join(DATA_DIR, dir_name, 'movies-companies.csv')
  elif data_name == 'book-crossing':
    dir_name = 'BX-filtered-authors'
    filepath = os.path.join(DATA_DIR, dir_name, 'BX-book-authors.csv')
  f = open(filepath, 'a', newline='', encoding = 'utf-8')
  w = csv.writer(f, lineterminator = '\n')
  for id in company_ids:
    result = [movie_id, id]
    print(result)
    w.writerow(result)
  f.close()

def write_company_to_file(data_name, company_id, company_name):
  if data_name == 'movie-lens-small':
    dir_name = 'ml-latest-small-company'
    filepath = os.path.join(DATA_DIR, dir_name, 'companies.csv')
  elif data_name == 'movie-lens-25m':
    dir_name = 'ml-25m-company'
    filepath = os.path.join(DATA_DIR, dir_name, 'companies.csv')
  elif data_name == 'book-crossing':
    dir_name = 'BX-filtered-authors'
    filepath = os.path.join(DATA_DIR, dir_name, 'BX-authors.csv')
  f = open(filepath, 'a', newline='', encoding = 'utf-8')
  w = csv.writer(f, lineterminator = '\n')
  result = [company_id, company_name]
  print(result)
  w.writerow(result)
  f.close()

def write_skipped_movies_to_file(skipped):
  dir_name = 'ml-25m-company'
  filepath = os.path.join(DATA_DIR, dir_name, 'skipped-movies.csv')
  f = open(filepath, 'a', newline='', encoding = 'utf-8')
  w = csv.writer(f, lineterminator = '\n')
  for movieId in zip(skipped):
    result = movieId
    w.writerow(result)
  f.close()

def get_last_line(filepath):
  last_line = ''
  with open(filepath, 'r', newline='', encoding = 'utf-8') as f:
    f.seek(-2, os.SEEK_END)
    while f.read(1) != b'\n':
      f.seek(-2, os.SEEK_CUR)
    last_line = f.readline()
  return last_line.split(',')
  
def read_companies(name):
  """ """
  if name == 'movie-lens-small':
    dir_name = 'ml-latest-small-company'
  elif name == 'movie-lens-25m':
    dir_name = 'ml-25m-company'
  csv_params = dict(header=0, usecols=[0, 1], names=['companyId', 'companyName'])
  company_data = read_csv(dir_name, 'companies.csv', csv_params)
  companies = {}
  n = 0
  for k, v in zip(company_data['companyName'], company_data['companyId']):
    companies[k] = v
    n = n + 1
  return companies, n

def get_movie_company_data(name):

  if name == 'movie-lens-25m':
    dir_name = 'ml-25m/ml-25m'
  elif name == 'movie-lens-small':
    dir_name = 'ml-latest-small'

  csv_params = dict(header=0, usecols=[0, 2], names=['movieId', 'tmdbId'])
  tmdb_data = read_csv(dir_name, 'links.csv', csv_params)
  tmdb_data = tmdb_data[tmdb_data['movieId'] >= 205431]
  tmdb_data['movieId'] = tmdb_data['movieId'].apply(lambda x: f'{x:.0f}').astype(str)
  tmdb_data['tmdbId'] = tmdb_data['tmdbId'].apply(lambda x: f'{x:.0f}').astype(str)
  
  companies, n = read_companies(name)

  # movies skipped because tmdb info does not exist
  for movie_id, tmdb_id in zip(tmdb_data['movieId'], tmdb_data['tmdbId']):
    print('movie_id: ' + movie_id)
    print('tmdb_id: ' + tmdb_id)
    if tmdb_id != 'nan':
      company_lst, company_ids, n, companies = get_companies(tmdb_id, n, companies)
      if company_lst != None:
        print(company_lst, company_ids)
        write_items_companies_to_file(name, company_ids, movie_id)

  skipped = tmdb_data[tmdb_data['tmdbId'] == 'nan']['movieId']
  write_skipped_movies_to_file(skipped)

  # movie_id = '5041'
  # tmdb_id = '15035'
  # print('movie_id: ' + movie_id)
  # print('tmdb_id: ' + tmdb_id)
  # company_lst, company_ids, n, companies = get_companies(tmdb_id, n, companies)
  # if company_lst != None:
  #   print(company_lst, company_ids)
  #   write_items_companies_to_file(company_ids, movie_id)

def get_publisher_data(publisher_data):
  publishers = {}
  n=0
  for publisher in publisher_data:
    if publisher not in publishers:
      n = n+1
      publishers[publisher] = n
      write_company_to_file('book-crossing', n, publisher)

#99347 authors reduced to 97363 authors
def get_author_data(temp_data):
  authors = {}
  n=0
  # [for book_id, author_name in zip(temp_data['bookId'],temp_data['producerName'])]
  for book_id, author_name in zip(temp_data['itemId'],temp_data['producerName']):
    # print(author_name)
    try:
      author = unicodedata.normalize('NFKD', author_name.casefold())
    except:
      author = 'Not Available'
    if author == '"Denning &amp; Phillips"':
      author = "Melita Denning"
    elif author == '"Lisa &amp; Diane Berger"':
      author = "Diane Berger"
    else:
      titles1 = ["Ph. D", "Ph.D", "M. D", "M.D", "D.V.M"]
      special_strs = ['"',",", "."]
      titles2 = [" Md ", " Dr ", " Phd ", " Ms ", " Dvm ", " Mrs ", " Jr ", " Prof "]
      extra_strs = titles1 + special_strs + titles2
      for str in extra_strs:
        author = author.replace(str, ' ')
      if author[0:4] == 'Not ' or author[0:7] == 'Various':
        author = 'Not Available'
      else:
        words = author.split()
        words = [word + '.' if len(word) == 1 else word for word in words]
        author = ' '.join(words)
    
    if author not in authors:
      n = n+1
      authors[author] = n
      # write_company_to_file('book-crossing', n, author.title())
    write_items_companies_to_file('book-crossing',[n],book_id)

def get_rating_data(track_data):
  return (track_data.groupby(['userId','producerName', 'trackName'])
          .size().reset_index(name='playCounts'))

def get_production_data(track_data):
  return None

def get_users(rating_data):
  return rating_data['userId'].drop_duplicates()

def column_to_string(dataset, column):
  return dataset[column].apply(lambda x: f'{x:.0f}').astype(str)

def get_data(name):
  """Read a dataset specified by name into pandas dataframe."""

  if name[0:5] == 'movie':
    if name == 'movie-lens-25m':
      dir_name = 'ml-25m/ml-25m'
      company_dir_name = 'ml-25m-company'
    elif name == 'movie-lens-small':
      dir_name = 'ml-latest-small'
      company_dir_name = 'ml-latest-small-company'
    csv_params = dict(header=0, usecols=[0, 1], names=['itemId', 'itemName'])
    item_data = read_csv(dir_name, 'movies.csv', csv_params)

    csv_params = dict(header=0, usecols=[0, 1, 2], 
      names=['userId', 'itemId', 'rating',])
    rating_data = read_csv(dir_name, 'ratings.csv', csv_params)
    rating_data['itemId'] = rating_data['itemId'].apply(lambda x: f'{x:.0f}').astype(str)
    rating_data['userId'] = rating_data['userId'].apply(lambda x: f'{x:.0f}').astype(str)
    
    user_data = get_users(rating_data) 

    csv_params = dict(header=0, usecols=[0, 1], names=['producerId', 'producerName'])
    producer_data = read_csv(company_dir_name, 'companies.csv', csv_params)
    producer_data['producerId'] = column_to_string(producer_data, 'producerId')

    csv_params = dict(header=0, usecols=[0, 1], names=['itemId', 'producerId']) 
    production_data = read_csv(company_dir_name, 'movies-companies.csv', csv_params)
    production_data['itemId'] = column_to_string(production_data, 'itemId')
    production_data['producerId'] = column_to_string(production_data, 'producerId')

    data = (producer_data, item_data, user_data, rating_data, production_data)
  elif name == 'book-crossing':
    dir_name = 'BX-CSV'
    csv_params = dict(header=0, usecols=[0, 1, 2], names=['itemId', 'title', 'tempName'],
      quotechar='"', delimiter=';',quoting=csv.QUOTE_ALL, skipinitialspace=True,
      escapechar='\\', engine='python')
    book_data = read_csv(dir_name, 'BX-Books.csv', csv_params)

    csv_params = dict(header=0, usecols=[0, 1, 2], 
      names=['userId', 'itemId', 'rating'],
      quotechar='"', delimiter=';',quoting=csv.QUOTE_ALL,skipinitialspace=True,
      escapechar='\\', engine='python')
    ratings = read_csv(dir_name, 'BX-Book-Ratings.csv', csv_params)
    ratings['rating'] = ratings['rating'].apply(lambda x: x/2)
    ratings['userId'] = column_to_string(ratings, 'userId')
    
    merged_data = pd.merge(book_data, ratings, on = 'itemId', how = 'inner')
    print(merged_data.head())
    
    items = merged_data[['itemId', 'title', 'tempName']].drop_duplicates().reset_index(drop = True)
    temp_data = items[['itemId','tempName']]

    # only run when author data is empty
    # get_author_data(temp_data)
    csv_params = dict(header=0,usecols=[0, 1],names=['producerId','producerName'],
      engine='python')

    producer_data = read_csv('BX-filtered-authors', 'BX-authors.csv',csv_params)
    producer_data['producerId'] = column_to_string(producer_data, 'producerId')

    csv_params = dict(header=0,usecols=[0, 1],names=['itemId','producerId'],
      engine='python')
    production_data = read_csv('BX-filtered-authors', 'BX-book-authors.csv',csv_params)
    production_data['producerId'] = column_to_string(production_data, 'producerId')
    # print('production df:\t{}'.format(production_data.shape))
    # print(production_data.head())

    min_book_ratings = 2
    item_rating_producer = pd.merge(production_data, merged_data)
    min_ratings_filter = item_rating_producer['itemId'].value_counts() > min_book_ratings
    filtered_data = item_rating_producer[item_rating_producer['itemId'].map(min_ratings_filter)]

    rating_data = filtered_data[['userId', 'itemId', 'rating']]
    
    print('item-rating-producer df:\t{}'.format(item_rating_producer.shape))
    print(item_rating_producer.head())

    item_data = filtered_data[['itemId', 'title']].drop_duplicates().reset_index(drop=True)
    user_data = get_users(rating_data)
    production_data = filtered_data[['itemId', 'producerId']].drop_duplicates().reset_index(drop=True)
    producer_data = None

    data = (producer_data, item_data, user_data, rating_data, production_data)
  else:
    ValueError('not a valid dataset name')
    
  return data

