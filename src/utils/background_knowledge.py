import numpy as np
import pandas as pd
import seaborn as sns


def draw_observ_prob_distribution():
    """ plot the curves of prob distribution """
    def data_prepare(std_deviation = 20):
        observ_prob_factor = 1 / (np.sqrt(2*np.pi) * std_deviation)

        def helper(x):
            return observ_prob_factor * np.exp(-np.power(x, 2)/(2*np.power(std_deviation, 2)))

        df = pd.DataFrame({'x': np.arange(0, 100, .1)})
        df.loc[:, 'y'] = df.x.apply( helper)
        df.loc[:, '_std'] = ""+ str(std_deviation)
        
        return df

    df = pd.concat( [ data_prepare(i) for i in range(5, 30, 5) ] )

    ax = sns.lineplot(x=df.x, y= df.y, hue=df._std)