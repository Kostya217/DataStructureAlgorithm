import pandas as pd
import numpy as np


def get_data():
    # Read in data
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    movies = pd.read_csv('ml-latest-small/movies.csv')

    # Merge ratings and movies dataset
    df = pd.merge(ratings, movies, on='movieId', how='inner')

    # Aggregate by movie
    agg_ratings = df.groupby('title') \
        .agg(mean_rating=('rating', 'mean'),
             number_of_ratings=('rating', 'count')).reset_index()

    # Keep the movies with over 100 ratings
    agg_ratings_gt100 = agg_ratings[agg_ratings['number_of_ratings'] > 100]

    # Return merged data
    return pd.merge(df, agg_ratings_gt100[['title']], on='title', how='inner')


def user_based_rec(picked_userid=1, number_of_similar_user=10, number_of_recommendations=3):
    # 1. Find similar users based on interactions with common item.
    # 2. Identify the items rated high by similar users but have not been exposed to the active user of interest.
    # 3. Calculate the weighted average score for each item.
    # 4. Rank item based on the score and pick the top n items of recommend.

    # Marge data
    df_gt100 = get_data()

    # Create user-item matrix
    matrix = df_gt100.pivot_table(
        index='userId',
        columns='title',
        values='rating')

    # Normalize user-item matrix
    matrix_norm = matrix.subtract(
        matrix.mean(axis=1),
        axis='rows')

    # User similarity matrix using Pearson correlation
    user_similarity = matrix_norm.T.corr()

    # Remove picked user ID from the candidate list
    user_similarity.drop(index=picked_userid, inplace=True)

    # User similarity threshold
    user_similarity_threshold = 0.3

    # Get top n similar users
    similar_users = user_similarity[user_similarity[picked_userid] > user_similarity_threshold][
                        picked_userid].sort_values(ascending=False)[:number_of_similar_user]

    # Print out top n similar users
    # print(f'The similar users for user {picked_userid} are', similar_users)

    # Movies that the target user has watched
    picked_userid_watched = matrix_norm[matrix_norm.index == picked_userid].dropna(axis=1, how='all')

    # Movies that similar users watched. Remove movies that none of the similar users have watched
    similar_user_movies = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')

    # Remove the watched movie from the movie list
    similar_user_movies.drop(picked_userid_watched.columns, axis=1, inplace=True, errors='ignore')

    # A dictionary to store item scores
    item_score = {}
    # Loop through items
    for i in similar_user_movies.columns:
        # Get the ratings for movie i
        movie_rating = similar_user_movies[i]
        # Create a variable to store the score
        total = 0
        # Create a variable to store the number of scores
        count = 0
        # Loop through similar users
        for u in similar_users.index:
            # If the movie has rating
            if pd.isna(movie_rating[u]) is False:
                # Score is the sum of user similarity score multiply by the movie rating
                score = similar_users[u] * movie_rating[u]
                # Add the score to the total score for the movie so far
                total += score
                # Add 1 to the count
                count += 1
        # Get the average score for the item
        item_score[i] = total / count
    # Convert dictionary to pandas dataframe
    item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])

    # Sort the movies by score
    ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)

    # Select top m movies
    print(ranked_item_score.head(number_of_recommendations))


def item_based_rec(picked_userid=1, number_of_similar_items=5, number_of_recommendations=3):
    # 1. Calculate item similarity scores based on all the user ratings.
    # 2. Identify the top n item that are the most similar item of interest.
    # 3. Calculated the weight average score for the most similar item by the user.
    # 4. Rank items based on the score and pick top n items to recommend.

    # Merge data
    df_gt100 = get_data()

    # Create user-item matrix
    matrix = df_gt100.pivot_table(index='title', columns='userId', values='rating')

    # Normalize user-item matrix
    matrix_norm = matrix.subtract(matrix.mean(axis=1), axis=0)

    # Item similarity matrix using Pearson correlation
    item_similarity = matrix_norm.T.corr()

    # Item-based recommendation function
    import operator
    # Movies that the target user has not watched
    picked_userid_unwatched = pd.DataFrame(matrix_norm[picked_userid].isna()).reset_index()
    picked_userid_unwatched = picked_userid_unwatched[picked_userid_unwatched[1] == True]['title'].values.tolist()

    # Movies that the target user has watched
    picked_userid_watched = pd.DataFrame(
        matrix_norm[picked_userid].
            dropna(axis=0, how='all') \
            .sort_values(ascending=False)).reset_index().rename(columns={1: 'rating'})

    # Dictionary to save the unwatched movie and predicted rating pair
    rating_prediction = {}

    # Loop through unwatched movies
    for picked_movie in picked_userid_unwatched:
        # Calculate the similarity score of the picked movie with other movies
        picked_movie_similarity_score = item_similarity[[picked_movie]].reset_index().rename(
            columns={picked_movie: 'similarity_score'})
        # Rank the similarities between the picked user watched movie and the picked unwatched movie.
        picked_userid_watched_similarity = pd.merge(
            left=picked_userid_watched,
            right=picked_movie_similarity_score,
            on='title',
            how='inner').sort_values('similarity_score', ascending=False)[:number_of_similar_items]

        # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
        predicted_rating = round(np.average(picked_userid_watched_similarity['rating'],
                                            weights=picked_userid_watched_similarity['similarity_score']), 6)
        # Save the predicted rating in the dictionary
        rating_prediction[picked_movie] = predicted_rating

        # Return the top recommended movies
    return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]


def main():
    print('User-based: ')
    user_based_rec(picked_userid=1, number_of_similar_user=10, number_of_recommendations=10)
    print()

    print('Item-based: ')
    # Get recommendations
    recommended_movie = item_based_rec(picked_userid=1, number_of_similar_items=5, number_of_recommendations=10)
    print(*recommended_movie, sep='\n')


if __name__ == '__main__':
    main()
