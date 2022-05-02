
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import scipy
import logging


logger = logging.getLogger(__name__)


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


def save_model(model, name):
    pd.to_pickle(model, f"{name}")


def load_model(name):
    path = f"{name}.pkl"
    return pd.read_pickle(f"{name}")


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)



def get_feature_neutral_mean(df, prediction_col, target_col):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [prediction_col],
                                          feature_cols)[prediction_col]
    scores = df.groupby("era").apply(
        lambda x: (unif(x["neutral_sub"]).corr(x[target_col]))).mean()
    return np.mean(scores)


def fast_score_by_date(df, columns, target, tb=None, era_col="era"):
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        era_pred = np.float64(df_era[columns].values.T)
        era_target = np.float64(df_era[target].values.T)

        if tb is None:
            ccs = np.corrcoef(era_target, era_pred)[0, 1:]
        else:
            tbidx = np.argsort(era_pred, axis=1)
            tbidx = np.concatenate([tbidx[:, :tb], tbidx[:, -tb:]], axis=1)
            ccs = [np.corrcoef(era_target[tmpidx], tmppred[tmpidx])[0, 1] for tmpidx, tmppred in zip(tbidx, era_pred)]
            ccs = np.array(ccs)

        computed.append(ccs)

    return pd.DataFrame(np.array(computed), columns=columns, index=df[era_col].unique())


def validation_metrics(validation_data, pred_cols, example_col, era_col, target_col, fast_mode=False):
    validation_stats = pd.DataFrame()
    feature_cols = [c for c in validation_data if c.startswith("feature_")]
    for pred_col in pred_cols:
        # Check the per-era correlations on the validation set (out of sample)
        validation_correlations = validation_data.groupby(era_col).apply(
            lambda d: unif(d[pred_col]).corr(d[target_col]))

        mean = validation_correlations.mean()
        std = validation_correlations.std(ddof=0)
        sharpe = mean / std

        validation_stats.loc["mean", pred_col] = mean
        validation_stats.loc["std", pred_col] = std
        validation_stats.loc["sharpe", pred_col] = sharpe

        rolling_max = (validation_correlations + 1).cumprod().rolling(window=9000,  # arbitrarily large
                                                                      min_periods=1).max()
        daily_value = (validation_correlations + 1).cumprod()
        max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
        validation_stats.loc["max_drawdown", pred_col] = max_drawdown

        payout_scores = validation_correlations.clip(-0.25, 0.25)
        payout_daily_value = (payout_scores + 1).cumprod()

        apy = (
            (
                (payout_daily_value.dropna().iloc[-1])
                ** (1 / len(payout_scores))
            )
            ** 49  # 52 weeks of compounding minus 3 for stake compounding lag
            - 1
        ) * 100

        validation_stats.loc["apy", pred_col] = apy

        if not fast_mode:
            # Check the feature exposure of your validation predictions
            max_per_era = validation_data.groupby(era_col).apply(
                lambda d: d[feature_cols].corrwith(d[pred_col]).abs().max())
            max_feature_exposure = max_per_era.mean()
            validation_stats.loc["max_feature_exposure", pred_col] = max_feature_exposure

            # Check feature neutral mean
            feature_neutral_mean = get_feature_neutral_mean(validation_data, pred_col, target_col)
            validation_stats.loc["feature_neutral_mean", pred_col] = feature_neutral_mean

            # Check top and bottom 200 metrics (TB200)
            tb200_validation_correlations = fast_score_by_date(
                validation_data,
                [pred_col],
                target_col,
                tb=200,
                era_col=era_col
            )

            tb200_mean = tb200_validation_correlations.mean()[pred_col]
            tb200_std = tb200_validation_correlations.std(ddof=0)[pred_col]
            tb200_sharpe = tb200_mean / tb200_std

            validation_stats.loc["tb200_mean", pred_col] = tb200_mean
            validation_stats.loc["tb200_std", pred_col] = tb200_std
            validation_stats.loc["tb200_sharpe", pred_col] = tb200_sharpe

        # MMC over validation
        mmc_scores = []
        corr_scores = []
        for _, x in validation_data.groupby(era_col):
            series = neutralize_series(unif(x[pred_col]), (x[example_col]))
            mmc_scores.append(np.cov(series, x[target_col])[0, 1] / (0.29 ** 2))
            corr_scores.append(unif(x[pred_col]).corr(x[target_col]))

        val_mmc_mean = np.mean(mmc_scores)
        val_mmc_std = np.std(mmc_scores)
        corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
        corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)

        validation_stats.loc["mmc_mean", pred_col] = val_mmc_mean
        validation_stats.loc["corr_plus_mmc_sharpe", pred_col] = corr_plus_mmc_sharpe

        # Check correlation with example predictions
        per_era_corrs = validation_data.groupby(era_col).apply(lambda d: unif(d[pred_col]).corr(unif(d[example_col])))
        corr_with_example_preds = per_era_corrs.mean()
        validation_stats.loc["corr_with_example_preds", pred_col] = corr_with_example_preds

    # .transpose so that stats are columns and the model_name is the row
    return validation_stats.transpose()



def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def get_columns(filename: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    all_columns = get_all_columns(filename)
    features = [c for c in all_columns if c.startswith("feature_")]
    targets = [c for c in all_columns if c.startswith("target_") or c == 'target']
    other_columns = [c for c in all_columns if c != 'target' and not c.startswith("feature_") and not c.startswith("target_")]
    # logger.info(f"Columns loaded.\nFeatures: {len(features)}\nTargets: {len(targets)}\nOther: {len(other_columns)}")
    print(pd.DataFrame([{
        "features": len(features),
        "targets": len(targets),
        "other": len(other_columns),
        "stats": "nb_cols"
    }]).set_index("stats"))
    return all_columns, features, targets, other_columns