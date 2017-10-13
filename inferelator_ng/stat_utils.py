import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def compute_stats(dataframe):
    return (dataframe.mean(axis=1), dataframe.var(axis=1))

def normalize(df, mean, sigma_squared):
    return df.sub(mean, axis=0).div(np.sqrt(sigma_squared), axis = 0)

def filter_out(df, filter_list):
    return df[df.columns.difference(filter_list)]

def compute_error(X, Y, thresholded_matrix, held_out_X, held_out_Y):
    """
    Does a linear fit using sklearn, on the non-zero predictors. 
    Returns the mean squared error on training data per gene and test data per gene
    We will normalize the testdata and the testactivity with the mean and std of the train, as described in
    https://stats.stackexchange.com/questions/174823/how-to-apply-standardization-normalization-to-train-and-testset-if-prediction-i
    """

    # 8/18/17: try to debug NAs
    def drop_nas(df):
        prev_shape = df.shape
        new_df = df.dropna(axis=1, how='all')
        return new_df

    test_error = {'counts':{}}
    train_error = {'counts':{}}

    held_out_X = drop_nas(held_out_X)
    held_out_Y = drop_nas(held_out_Y)

    X = drop_nas(X)
    Y = drop_nas(Y)

    (X_mu, X_sigma_squared) = compute_stats(X)
    X_normalized = normalize(X, X_mu, X_sigma_squared)
    (Y_mu, Y_sigma_squared) = compute_stats(Y)
    Y_normalized = normalize(Y, Y_mu, Y_sigma_squared)

    held_out_Y_normalized = normalize(held_out_Y, Y_mu, Y_sigma_squared)
    held_out_X_normalized = normalize(held_out_X, X_mu, X_sigma_squared)
    ols = LinearRegression(normalize=False, fit_intercept=False)
    for gene_name, y_normalized in Y_normalized.iterrows():
        nonzero  = thresholded_matrix.loc[gene_name,:].nonzero()[0]
        #only compute betas if there was found to be a predictive TF for this target gene
        if len(nonzero) > 0:
            nonzero_X_normalized = X_normalized.iloc[nonzero,:].transpose()
            n = len(y_normalized)
            fitted = ols.fit(nonzero_X_normalized, y_normalized)
            train_error[gene_name] = np.sum((ols.predict(nonzero_X_normalized) - y_normalized) ** 2) 
            train_error['counts'][gene_name] = n
            # fig = sm.graphics.plot_regress_exog(ols, tf, fig=fig)
            held_out_nonzero_X_normalized = held_out_X_normalized.iloc[nonzero,:].transpose()
            n = len(held_out_Y_normalized.loc[gene_name,:])
            test_error[gene_name] = np.sum((ols.predict(held_out_nonzero_X_normalized) - held_out_Y_normalized.loc[gene_name,:]) ** 2) 
            test_error['counts'][gene_name] = n
        else:
            # predicted ols solution is mean of Y, i.e. Y_mu
            train_error[gene_name] = np.sum(Y_normalized ** 2) 
            train_error['counts'][gene_name] = n
            test_error[gene_name] = np.sum(held_out_Y_normalized ** 2)
            test_error['counts'][gene_name] = n
            
    return (train_error, test_error)

def plot_tf(gene, x, y, fittedvalues, tfs, plot_type = 'train'):
    for tf in tfs:
        plt.plot(x[tf], y, 'ro')
        plt.plot(x[tf], fittedvalues, 'bo')
        plt.legend(['Data', 'Fitted model'])
        plt.xlabel(tf)
        plt.savefig("{}_{}_{}_plt.png".format(gene, tf, plot_type))
        plt.clf()

