if __name__ == '__main__':
    import sys
    sys.path.append('D:/projects/numerai-tournament') 

from typing import List, Literal, Optional, Tuple, Union
from dotenv import load_dotenv
from tqdm import tqdm
from models.utils import create_folder
load_dotenv()
from models.utils import get_all_columns, get_biggest_change_features, load_model, neutralize, save_model
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


def get_training_data(filename: str, features_set: str='full', features_file=None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    all_columns, features, targets, other_cols = get_columns(filename)

    logger.info('Reading minimal training data')
    # read the feature metadata and get the "small" feature set
    
    if features_set != 'full' and features_file is not None:
        with open(features_file, "r") as f:
            feature_metadata = json.load(f)
        features = feature_metadata["feature_sets"][features_set]
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


def get_model(training_data, target_col, filename:Optional[str]=None):
    if filename is not None and os.path.exists(filename):
        return load_model(filename)
    params = {"n_estimators": 2000,
                "learning_rate": 0.01,
                "max_depth": 5,
                "num_leaves": 2 ** 5,
                "colsample_bytree": 0.1}
    model = LGBMRegressor(**params)

    # train on all of train and save the model so we don't have to train next time
    model.fit(training_data.filter(like='feature_', axis='columns'), training_data[target_col])
    logger.info(f"saving new model: {filename}")
    save_model(model, filename)
    return model


def run(features_set: Literal['full', 'small', 'medium', 'legacy']="full", force_regen: bool=True, neutralize_riskiest: Optional[Union[List[int], int]]=None):
    if neutralize_riskiest is None:
        neutralize_riskiest = 168
    public_id = os.environ.get("NUMERAI_PUBLIC_KEY")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")
    
    logger.info("Connecting to Numerai server")
    napi = numerapi.NumerAPI(public_id, secret_key)
    logger.info("Successfully logged into Numerai API")

    current_round = napi.get_current_round()
    logger.info(f"Current Round : {current_round}")
    
    DATA_FOLDER = "data"
    create_folder(DATA_FOLDER)
    ROUND_FOLDER = os.path.join(DATA_FOLDER, str(current_round))
    create_folder(ROUND_FOLDER)
    INPUT_FOLDER = os.path.join(ROUND_FOLDER, "input")
    create_folder(INPUT_FOLDER)
    OUTPUT_FOLDER = os.path.join(ROUND_FOLDER, "output", "lgbm", features_set)
    create_folder(OUTPUT_FOLDER)
    MODEL_FOLDER = os.path.join(ROUND_FOLDER, 'cache', features_set)
    create_folder(MODEL_FOLDER)
    
    TRAINING_DATA_FILE = os.path.join(INPUT_FOLDER, "training_data.parquet")
    TOURNAMENT_DATA_FILE = os.path.join(INPUT_FOLDER, f"tournament_data.parquet")
    VALIDATION_DATA_FILE = os.path.join(INPUT_FOLDER, "validation_data.parquet")
    EXAMPLE_VALIDATION_PREDICTIONS_FILE = os.path.join(INPUT_FOLDER, "example_validation_predictions.parquet")
    FEATURES_FILE = os.path.join(INPUT_FOLDER, "features.json")

    MODEL_NAME = "lbgm"
    TARGET_MODEL_FILE = os.path.join(MODEL_FOLDER, f"{MODEL_NAME}.model")
    FEATURES_CORR_FILE = os.path.join(MODEL_FOLDER, "features_corr.csv")
    VALIDATION_PREDICTIONS_FILE = os.path.join(OUTPUT_FOLDER, f"validation_predictions.csv")
    TOURNAMENT_PREDICTIONS_FILE = os.path.join(OUTPUT_FOLDER, f"tournament_predictions.csv")

    PREDICTION_COL = f"prediction"
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
    
    training_data, all_columns, features = get_training_data(TRAINING_DATA_FILE, features_set, FEATURES_FILE)
    
    model = get_model(training_data, TARGET_COL, TARGET_MODEL_FILE)
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
    
    validation_data.loc[:, PREDICTION_COL] = model.predict(
        validation_data.loc[:, model_expected_features])
    tournament_data.loc[:, PREDICTION_COL] = model.predict(
        tournament_data.loc[:, model_expected_features])

    gc.collect()
    
    if force_regen or not os.path.exists(FEATURES_CORR_FILE):
        # getting the per era correlation of each feature vs the target
        all_feature_corrs = training_data.groupby(ERA_COL).apply(
            lambda era: era[features].corrwith(era[TARGET_COL])
        )
        all_feature_corrs.to_csv(FEATURES_CORR_FILE)
    else:
        all_feature_corrs = pd.read_csv(FEATURES_CORR_FILE)
    
    if type(neutralize_riskiest) is int:
        neutralize_riskiest = [neutralize_riskiest]
    
    if type(neutralize_riskiest) is list:  
        for n in tqdm(neutralize_riskiest):
            validation_file = os.path.join(OUTPUT_FOLDER, f"validation_predictions_{n}.csv")
            tournament_file = os.path.join(OUTPUT_FOLDER, f"tournament_predictions_{n}.csv")
            riskiest_features = get_biggest_change_features(all_feature_corrs, n)
            if force_regen or not os.path.exists(validation_file):
                # neutralize our predictions to the riskiest features
                validation_result = neutralize(
                    df=validation_data,
                    columns=[PREDICTION_COL],
                    neutralizers=riskiest_features,
                    proportion=1.0,
                    normalize=True,
                    era_col=ERA_COL
                )
                validation_result = validation_result.rank(pct=True)
                validation_result.to_csv(validation_file)

            if force_regen or not os.path.exists(tournament_file):
                tournament_result = neutralize(
                    df=tournament_data,
                    columns=[PREDICTION_COL],
                    neutralizers=riskiest_features,
                    proportion=1.0,
                    normalize=True,
                    era_col=ERA_COL
                )
                
                tournament_result = tournament_result.rank(pct=True)
                tournament_result.to_csv(tournament_file)

            # rename best model to "prediction" and rank from 0 to 1 to meet upload requirements
            # validation_data["prediction"] = validation_data[model_to_submit].rank(pct=True)
            # tournament_data["prediction"] = tournament_data[model_to_submit].rank(pct=True)
            
            # save predictions to csv
            # validation_data["prediction"].to_csv(VALIDATION_PREDICTIONS_FILE)
            # tournament_data["prediction"].to_csv(TOURNAMENT_PREDICTIONS_FILE)
    
    # validation_preds = pd.read_parquet(EXAMPLE_VALIDATION_PREDICTIONS_FILE)
    # validation_data[EXAMPLE_PREDS_COL] = validation_preds["prediction"]
    
    # get some stats about each of our models to compare...
    # fast_mode=True so that we skip some of the stats that are slower to calculate
    # validation_stats = validation_metrics(validation_data, [model_to_submit], example_col=EXAMPLE_PREDS_COL, fast_mode=True)
    # print(validation_stats[["mean", "sharpe"]].to_markdown())

if __name__ == '__main__':
    from datetime import datetime
    start = datetime.now()
    values = [160 + x for x in range(10)]
    print(values)
    run(neutralize_riskiest=50, features_set='small')
    print((datetime.now() - start).total_seconds())
