#!/usr/bin/env python
# coding: utf-8

import numpy as np


def epsilon_greedy_policy(df, arms, epsilon=0.15, slate_size=5, batch_size=50):
    '''
    Applies Epsilon Greedy policy to generate movie recommendations.
    Args:
        df: dataframe. Dataset to apply the policy to
        arms: list or array. ID of every eligible arm.
        epsilon: float. represents the % of timesteps where we explore random arms
        slate_size: int. the number of recommendations to make at each step.
        batch_size: int. the number of users to serve these recommendations to before updating the bandit's policy.
    '''
    # draw a 0 or 1 from a binominal distribution, with epsilon% likelihood of drawing a 1
    explore = np.random.binomial(1, epsilon)

    # if explore: shuffle movies to choose a random set of recommendations
    if explore == 1 or df.shape[0] == 0:
        recs = np.random.choice(arms, size=(slate_size), replace=False)
    # if exploit: sort moveis by "like rate", recommend movies with the best performance so far
    else:
        scores = df[['movieId', 'liked']].groupby('movieId').agg({'liked': ['mean', 'count']})
        scores.columns = ['mean', 'count']
        scores['movieId'] = scores.index
        scores = scores.sort_values('mean', ascending=False)
        recs = scores.loc[scores.index[0:slate_size], 'movieId'].values
    return recs
