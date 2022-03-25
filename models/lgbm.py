from re import T
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()
from utils import get_all_columns, get_biggest_change_features, neutralize
import pandas as pd
import numerapi
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy
import json
from lightgbm import LGBMRegressor
import gc
import json
import logging

logger = logging.getLogger("LGBM")


def get_training_data(filename: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    all_columns, features, targets, other_cols = get_columns(filename)

    logger.info('Reading minimal training data')
    # read the feature metadata and get the "small" feature set
    # with open(FEATURES_FILE, "r") as f:
    #     feature_metadata = json.load(f)
    # features = feature_metadata["feature_sets"]["small"]
    # read in just those features along with era and target columns
    read_columns = features + targets + other_cols
    # note: sometimes when trying to read the downloaded data you get an error about invalid magic parquet bytes...
    # if so, delete the file and rerun the napi.download_dataset to fix the corrupted file
    return pd.read_parquet(filename, columns=read_columns), all_columns, features
    

def get_columns(filename: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    all_columns = get_all_columns(filename)
    features = [c for c in all_columns if c.startswith("feature_")]
    targets = [c for c in all_columns if c.startswith("target_") or c == 'target']
    other_columns = [c for c in all_columns if c != 'target' and not c.startswith("feature_") and not c.startswith("target_")]
    logger.info(f"Columns loaded.\nFeatures: {len(features)}\nTargets: {len(targets)}\nOther: {len(other_columns)}")
    print(pd.DataFrame([{
        "features": len(features),
        "targets": len(targets),
        "other": len(other_columns),
        "stats": "nb_cols"
    }]).set_index("stats"))
    return all_columns, features, targets, other_columns


def create_folder(folder):
    if os.path.exists(folder):
        return
    logger.info(f"Creating {folder} folder")
    os.makedirs(folder)


def run(neutralize_riskiest: int=50):
    public_id = os.environ.get("NUMERAI_PUBLIC_KEY")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")
    
    logger.info("Connecting to Numerai server")
    napi = numerapi.NumerAPI(public_id, secret_key)
    logger.info("Successfully logged into Numerai API")

    current_round = napi.get_current_round()
    logger.info(f"Current Round : {current_round}")
    
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = os.path.join("OUTPUT", str(current_round))
    create_folder(DATA_FOLDER)
    create_folder(OUTPUT_FOLDER)
    
    TRAINING_DATA_FILE = os.path.join(DATA_FOLDER, "training_data.parquet")
    TOURNAMENT_DATA_FILE = os.path.join(DATA_FOLDER, f"tournament_data_{current_round}.parquet")
    VALIDATION_DATA_FILE = os.path.join(DATA_FOLDER, "validation_data.parquet")
    EXAMPLE_VALIDATION_PREDICTIONS_FILE = os.path.join(DATA_FOLDER, "example_validation_predictions.parquet")
    FEATURES_FILE = os.path.join(DATA_FOLDER, "features.json")

    MODEL_NAME = "lbgm"
    TARGET_MODEL_FILE = os.path.join(OUTPUT_FOLDER, f"{MODEL_NAME}.model")
    VALIDATION_PREDICTIONS_FILE = os.path.join(OUTPUT_FOLDER, "validation_predictions.csv")
    TOURNAMENT_PREDICTIONS_FILE = os.path.join(OUTPUT_FOLDER, "tournament_predictions.csv")

    TARGET_COL = 'target'
    ERA_COL = 'era'
    DATA_TYPE_COL = 'data_type'
    EXAMPLE_PREDS_COL = "example_preds"
    
    # Tournament data changes every week so we specify the round in their name. Training
    # and validation data only change periodically, so no need to download them every time.
    logger.info('Downloading dataset files...')
    napi.download_dataset("numerai_training_data.parquet", TRAINING_DATA_FILE)
    napi.download_dataset("numerai_tournament_data.parquet", TOURNAMENT_DATA_FILE)
    napi.download_dataset("numerai_validation_data.parquet", VALIDATION_DATA_FILE)
    napi.download_dataset("example_validation_predictions.parquet", EXAMPLE_VALIDATION_PREDICTIONS_FILE)
    napi.download_dataset("features.json", FEATURES_FILE)
    logger.info("All files ready")
    
    training_data, all_columns, features = get_training_data(TRAINING_DATA_FILE)
    
    # getting the per era correlation of each feature vs the target
    all_feature_corrs = training_data.groupby(ERA_COL).apply(
        lambda era: era[features].corrwith(era[TARGET_COL])
    )
    
    riskiest_features = get_biggest_change_features(all_feature_corrs, neutralize_riskiest)
    
    # gc.collect()
    
    params = {"n_estimators": 2000,
                "learning_rate": 0.01,
                "max_depth": 5,
                "num_leaves": 2 ** 5,
                "colsample_bytree": 0.1}
    model = LGBMRegressor(**params)

    # train on all of train and save the model so we don't have to train next time
    model.fit(training_data.filter(like='feature_', axis='columns'), training_data[TARGET_COL])
    # print(f"saving new model: {TARGET_MODEL_FILE}")
    # save_model(model, TARGET_MODEL_FILE)
    gc.collect()
    
    validation_data = pd.read_parquet(VALIDATION_DATA_FILE, columns=all_columns)
    tournament_data = pd.read_parquet(TOURNAMENT_DATA_FILE, columns=all_columns)

    tournament_data_features_only = tournament_data[features + [DATA_TYPE_COL]]
    nans_per_col = tournament_data_features_only[tournament_data_features_only["data_type"] == "live"].isna().sum()
    del tournament_data_features_only
    # check for nans and fill nans
    if nans_per_col.any():
        total_rows = len(tournament_data[tournament_data[DATA_TYPE_COL] == "live"])
        logger.info(f"Number of nans per column this week: {nans_per_col[nans_per_col > 0]}")
        logger.info(f"out of {total_rows} total rows")
        logger.info(f"filling nans with 0.5")
        tournament_data.loc[:, features] = tournament_data.loc[:, features].fillna(0.5)
    else:
        logger.info("No nans in the features this week!")
    
    
    # double check the feature that the model expects vs what is available to prevent our
    # pipeline from failing if Numerai adds more data and we don't have time to retrain!
    model_expected_features = model.booster_.feature_name()
    if set(model_expected_features) != set(features):
        logger.warn(f"New features are available! Might want to retrain model {MODEL_NAME}.")
    validation_data.loc[:, f"preds_{MODEL_NAME}"] = model.predict(
        validation_data.loc[:, model_expected_features])
    tournament_data.loc[:, f"preds_{MODEL_NAME}"] = model.predict(
        tournament_data.loc[:, model_expected_features])

    gc.collect()
    
    # neutralize our predictions to the riskiest features
    validation_data[f"preds_{MODEL_NAME}_neutral_riskiest_50"] = neutralize(
        df=validation_data,
        columns=[f"preds_{MODEL_NAME}"],
        neutralizers=riskiest_features,
        proportion=1.0,
        normalize=True,
        era_col=ERA_COL
    )

    tournament_data[f"preds_{MODEL_NAME}_neutral_riskiest_50"] = neutralize(
        df=tournament_data,
        columns=[f"preds_{MODEL_NAME}"],
        neutralizers=riskiest_features,
        proportion=1.0,
        normalize=True,
        era_col=ERA_COL
    )


    model_to_submit = f"preds_{MODEL_NAME}_neutral_riskiest_50"

    # rename best model to "prediction" and rank from 0 to 1 to meet upload requirements
    validation_data["prediction"] = validation_data[model_to_submit].rank(pct=True)
    tournament_data["prediction"] = tournament_data[model_to_submit].rank(pct=True)
    
    # save predictions to csv
    validation_data["prediction"].to_csv(VALIDATION_PREDICTIONS_FILE)
    tournament_data["prediction"].to_csv(TOURNAMENT_PREDICTIONS_FILE)
    
    validation_preds = pd.read_parquet(EXAMPLE_VALIDATION_PREDICTIONS_FILE)
    validation_data[EXAMPLE_PREDS_COL] = validation_preds["prediction"]
    
    # get some stats about each of our models to compare...
    # fast_mode=True so that we skip some of the stats that are slower to calculate
    # validation_stats = validation_metrics(validation_data, [model_to_submit], example_col=EXAMPLE_PREDS_COL, fast_mode=True)
    # print(validation_stats[["mean", "sharpe"]].to_markdown())

if __name__ == '__main__':
    from datetime import datetime
    start = datetime.now()
    run()
    print((datetime.now() - start).total_seconds())
