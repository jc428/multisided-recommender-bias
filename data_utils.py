from multiprocessing.sharedctypes import Value
import os, requests, json, csv

import numpy as np
import pandas as pd

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
  """ Gets the names of production companies for a given movie. 
  
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
        write_company_to_file(idx, company)
    return company_lst, company_ids, n, companies
  return None, None, n, companies

  
def write_movies_companies_to_file(company_ids, movie_id):
  dir_name = 'ml-latest-small-company'
  filepath = os.path.join(DATA_DIR, dir_name, 'movies-companies.csv')
  f = open(filepath, 'a', newline='', encoding = 'utf-8')
  w = csv.writer(f, lineterminator = '\n')
  for id in company_ids:
    result = [movie_id, id]
    print (result)
    w.writerow(result)
  f.close()

def write_company_to_file(company_id, company_name):
  dir_name = 'ml-latest-small-company'
  filepath = os.path.join(DATA_DIR, dir_name, 'companies.csv')
  f = open(filepath, 'a', newline='', encoding = 'utf-8')
  w = csv.writer(f, lineterminator = '\n')
  result = [company_id, company_name]
  print (result)
  w.writerow(result)
  f.close()

def write_skipped_movies_to_file(skipped):
  dir_name = 'ml-latest-small-company'
  filepath = os.path.join(DATA_DIR, dir_name, 'skipped-movies.csv')
  f = open(filepath, 'a', newline='', encoding = 'utf-8')
  w = csv.writer(f, lineterminator = '\n')
  for movieId in skipped:
    result = movieId
  print (result)
  w.writerow(result)
  f.close()

def get_movie_company_data(name):
  """Read a dataset specified by name into pandas dataframe."""

  if name == 'movie-lens-25m':
    dir_name = 'ml-25m'
    data_name = 'movies.csv'
  elif name == 'movie-lens-small':
    dir_name = 'ml-latest-small'

  csv_params = dict(header=0, usecols=[0, 2], names=['movieId', 'tmdbId'])
  tmdb_data = read_csv(dir_name, 'links.csv', csv_params)
  tmdb_data['movieId'] = tmdb_data['movieId'].apply(lambda x: f'{x:.0f}').astype(str)
  tmdb_data['tmdbId'] = tmdb_data['tmdbId'].apply(lambda x: f'{x:.0f}').astype(str)
  
  companies = {}
  n = 0
  # movies skipped because tmdb info does not exist
  skipped = []
  for movie_id, tmdb_id in zip(tmdb_data['movieId'], tmdb_data['tmdbId']):
    print('movie_id: ' + movie_id)
    print('tmdb_id: ' + tmdb_id)
    if tmdb_id == 'nan':
      skipped.append(movie_id)
    else:
      company_lst, company_ids, n, companies = get_companies(tmdb_id, n, companies)
      if company_lst != None:
        print(company_lst, company_ids)
        write_movies_companies_to_file(company_ids, movie_id)

  write_skipped_movies_to_file(skipped)

  # movie_id = '5041'
  # tmdb_id = '15035'
  # print('movie_id: ' + movie_id)
  # print('tmdb_id: ' + tmdb_id)
  # company_lst, company_ids, n, companies = get_companies(tmdb_id, n, companies)
  # if company_lst != None:
  #   print(company_lst, company_ids)
  #   write_movies_companies_to_file(company_ids, movie_id)

def get_data(name):
  """Read a dataset specified by name into pandas dataframe."""

  if name == 'movie-lens-25m':
    dir_name = 'ml-25m'
  elif name == 'movie-lens-small':
    dir_name = 'ml-latest-small'
    csv_params = dict(header=0, usecols=[0, 1], names=['movieId', 'title'])
    movie_data = read_csv(dir_name, 'movies.csv', csv_params)

    csv_params = dict(header=0, usecols=[0, 1, 2, 3], 
      names=['userId', 'movieId', 'rating', 'timestamp'])
    rating_data = read_csv(dir_name, 'ratings.csv', csv_params)

    dir_name = 'ml-latest-small-company'
    csv_params = dict(header=0, usecols=[0, 1], names=['companyId', 'companyName'])
    company_data = read_csv(dir_name, 'companies.csv', csv_params)

    csv_params = dict(header=0, usecols=[0, 1], names=['movieId', 'companyId']) 
    production_data = read_csv(dir_name, 'movies-companies.csv', csv_params)

    data = (movie_data, rating_data, company_data, production_data)
  elif name == 'book-crossing':
    dir_name = 'BX-CSV'
    csv_params = dict(header=0, usecols=[0, 1], names=['ISBN', 'title'],
      quotechar='"', delimiter=';',quoting=csv.QUOTE_ALL, skipinitialspace=True,
      escapechar='\\', engine='python')
    book_data = read_csv(dir_name, 'BX-Books.csv', csv_params)

    csv_params = dict(header=0, usecols=[0, 1, 2], 
      names=['userId', 'ISBN', 'rating'],
      quotechar='"', delimiter=';',quoting=csv.QUOTE_ALL,skipinitialspace=True,
      escapechar='\\', engine='python')
    rating_data = read_csv(dir_name, 'BX-Book-Ratings.csv', csv_params)

    # csv_params = dict(header=0, usecols=[0, 1], names=['companyId', 'companyName'])
    # company_data = read_csv('dir_name', 'companies.csv', csv_params)
    company_data = None

    csv_params = dict(header=0, usecols=[0, 4], names=['ISBN', 'companyName'],
      quotechar='"', delimiter=';',quoting=csv.QUOTE_ALL,skipinitialspace=True,
      escapechar='\\', engine='python')
    production_data = read_csv(dir_name, 'BX-Books.csv', csv_params)

    data = (book_data, rating_data, company_data, production_data)
  else:
    ValueError('not a valid dataset name')
    
  return data

book_data, rating_data, company_data, production_data = get_data("book-crossing")
print(book_data.head())
print(rating_data.head())
print(production_data.head())
