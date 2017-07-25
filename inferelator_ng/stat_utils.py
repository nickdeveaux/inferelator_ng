import numpy as np

def compute_stats(dataframe):
    return (dataframe.mean(axis=1), dataframe.var(axis=1))

def normalize(df, mu, sigma_squared):
    return df.sub(mu, axis=0).div(np.sqrt(sigma_squared), axis = 0)

def filter_out(df, filter_list):
    return df[df.columns.difference(filter_list)]
