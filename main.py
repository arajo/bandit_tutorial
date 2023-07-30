import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt

from EpsilonGreedy import epsilon_greedy_policy
from UCB1 import ucb1_policy
from utils import score, preprocess_movie_data_20m

parser = argparse.ArgumentParser()
parser.add_argument("--run_type", "--run_type")

args = parser.parse_args()

if __name__ == '__main__':
    run_type = args.run_type

    ratings = pd.read_csv("../../data/ml-latest-small/ratings.csv")
    ratings['liked'] = ratings.rating.map(lambda x: 1 if x >= 4.5 else 0)

    # Load dataset
    df = preprocess_movie_data_20m(ratings)

    # Setting Args
    batch_size = 50
    verbose = True
    slate_size = 5
    batch_size = 50
    epsilon = 0.15
    ucb_scale = 2.0

    # initialize empty history
    # (offline eval means you can only add to histroy when rec mathces historic data)
    history = pd.DataFrame(data=None, columns=df.columns)
    history = history.astype({'movieId': 'int32', 'liked': 'float'})

    # to speed this up, retrain the bandit every batch_size time steps
    # this lets us measure batch_size actions against a slate of recommendations rather than generating
    #      recs at each time step. this seems like the only way to make it through a large dataset like
    #      this and get a meaningful sample size with offline/replay evaluation

    rewards = []
    max_time = df.shape[0]  # total number of ratings to evaluate using the bandit
    for t in range(max_time // batch_size):
        t = t * batch_size
        if t % 100000 == 0:
            if verbose:
                print(t)
        # choose which arm to pull
        # apply epsilon greedy policy to the historic dataset (all arm-pulls prior to the current step that passed the replay-filter)
        if run_type == 'epsilon':
            recs = epsilon_greedy_policy(df=history.loc[history.t <= t,], arms=df.movieId.unique(), epsilon=epsilon,
                                         slate_size=slate_size, batch_size=batch_size)
        elif run_type == 'ucb1':
            recs = ucb1_policy(df=history.loc[history.t < t,], t=t, slate_size=slate_size, ucb_scale=ucb_scale)
        else:
            print("Unknown run type")
            break

        history, action_score = score(history, df, t, batch_size, recs)
        if action_score is not None:
            action_score = action_score.liked.tolist()
            rewards.extend(action_score)

    cumulative_avg = np.cumsum(rewards) / np.linspace(1, len(rewards), len(rewards))
    plt.plot(pd.Series(rewards).rolling(200).mean(), label='epsilon')
    plt.plot(cumulative_avg, label='epsilon')
