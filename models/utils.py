
from typing import List
import numpy as np
import pandas as pd
import scipy


def get_all_columns(parquet_file: str) -> List[str]:
    from pyarrow.parquet import ParquetFile
    import pyarrow
    pf = ParquetFile(parquet_file) 
    first_ten_rows = next(pf.iter_batches(batch_size = 100)) 
    training_data = pyarrow.Table.from_batches([first_ten_rows]).to_pandas()
    return training_data.columns

# find the riskiest features by comparing their correlation vs
# the target in each half of training data; we'll use these later
def get_biggest_change_features(corrs, n=None):
    """Find the riskiest features by comparing their correlation vs the target in each half of training data; we'll use these later

    Args:
        corrs (_type_): _description_
        n (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    all_eras = corrs.index.sort_values()
    if n is None:
        n = len(corrs.columns) // 2
    h1_eras = all_eras[:len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2:]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n


def neutralize(df,
                columns,
                neutralizers=None,
                proportion=1.0,
                normalize=True,
                era_col="era"):
        if neutralizers is None:
            neutralizers = []
        unique_eras = df[era_col].unique()
        computed = []
        for u in unique_eras:
            df_era = df[df[era_col] == u]
            scores = df_era[columns].values
            if normalize:
                scores2 = []
                for x in scores.T:
                    x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                    x = scipy.stats.norm.ppf(x)
                    scores2.append(x)
                scores = np.array(scores2).T
            exposures = df_era[neutralizers].values

            scores -= proportion * exposures.dot(
                np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

            scores /= scores.std(ddof=0)

            computed.append(scores)

        return pd.DataFrame(np.concatenate(computed),
                            columns=columns,
                            index=df.index)

