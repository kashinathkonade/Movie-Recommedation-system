
# Import the usefull libraries

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

credits_ = pd.read_csv(r"C:\Users\kashinath konade\Desktop\Grow Intern for DS\Movie Recommedation system\tmdb_5000_credits.csv")

movies = pd.read_csv(r"C:\Users\kashinath konade\Desktop\Grow Intern for DS\Movie Recommedation system\tmdb_5000_movies.csv")




credits_.head()

movies.head()


print("Credits:",credits_.shape)

print("Movies Dataframe:",movies.shape)


# here we renamed the column
credits_column_renamed = credits_.rename(index=str, columns={"movie_id": "id"})

movies_merge = movies.merge(credits_column_renamed, on='id')

print(movies_merge.head())

movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])

print(movies_cleaned.head())
print(movies_cleaned.info())


print(movies_cleaned.head(1)['overview'])

print(movies_cleaned.head(1)['overview'])


#Content Based Recommendation System


movies_cleaned['overview'].fillna('', inplace=True)
tfv = TfidfVectorizer(min_df=3, max_features=None,
strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
ngram_range=(1, 3),
stop_words = 'english')

# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(movies_cleaned['overview'])
print(tfv_matrix)
print(tfv_matrix.shape)


# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
print(sig[0])


# Reverse mapping of indices and movie titles
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()
print(indices)
print(indices['Newlyweds'])
print(sig[4799])
print(list(enumerate(sig[indices['Newlyweds']])))
print(sorted(list(enumerate(sig[indices['Newlyweds']])), key=lambda x: x[1], reverse=True))


def give_recomendations(title, sig=sig):
# Get the index corresponding to original_title
    idx = indices[title]
# Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))
# Sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
# Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]
# Movie indices
    movie_indices = [i[0] for i in sig_scores]
# Top 10 most similar movies
    return movies_cleaned['original_title'].iloc[movie_indices]


print(give_recomendations('Avatar'))
print(give_recomendations('Spectre'))


