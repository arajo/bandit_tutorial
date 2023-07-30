import numpy as np


def ucb1_policy(df, t, slate_size, ucb_scale=2.0):
    """
    Applies UCB1 policy to generate movie recommendations
    Args:
        df: dataframe. Dataset to apply UCB policy to.
        ucb_scale: float. Most implementations use 2.0
        t: int. represents the current time step.
    """

    scores = df[['movieId', 'liked']].groupby('movieId').agg({'liked': ['mean', 'count', 'std']})
    scores.columns = ['mean', 'count', 'std']
    scores['ucb'] = scores['mean'] + np.sqrt(
        (
                (2 * np.log10(t)) /
                scores['count']
        )
    )
    scores['movieId'] = scores.index
    scores = scores.sort_values('ucb', ascending=False)
    recs = scores.loc[scores.index[0:slate_size], 'movieId'].values
    return recs
