import numpy as np
import pandas as pd


def score(history, df, t, batch_size, recs):
    # replay score. reward if rec matches logged data, ignore otherwise
    actions = df[t:t + batch_size]
    actions = actions[actions['movieId'].isin(recs)]
    actions['scoring_round'] = t
    # add row to history if recs match logging policy
    history = pd.concat([history, actions], axis=0)
    action_liked = actions[['movieId', 'liked']]
    return history, action_liked


def preprocess_movie_data_20m(logs, min_number_of_reviews=20, balanced_classes=False):
    print('preparing ratings log')
    # remove ratings of movies with < N ratings. too few ratings will cause the recsys to get stuck in offline evaluation
    count_df = pd.DataFrame(logs.movieId.value_counts())
    print(f"{count_df.max()=} {count_df.min()=}")
    movies_to_keep = count_df.loc[count_df['count'] >= min_number_of_reviews].index
    logs = logs[logs['movieId'].isin(movies_to_keep)]

    if balanced_classes is True:
        logs = logs.groupby('movieId')
        logs = logs.apply(lambda x: x.sample(logs.size().min()).reset_index(drop=True))
    # shuffle rows to deibas order of user ids
    logs = logs.sample(frac=1)
    # create a 't' column to represent time steps for the bandit to simulate a live learning scenario
    logs['t'] = np.arange(len(logs))
    logs.index = logs['t']
    logs['liked'] = logs['rating'].apply(lambda x: 1 if x >= 4.5 else 0)
    return logs
