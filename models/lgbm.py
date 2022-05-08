from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
import os
if __name__ == '__main__':
    import sys
    project_root = os.environ.get('PROJECT_ROOT_DIR', '')
    sys.path.append(project_root) 

from typing import List, Literal, Optional
from models.utils import create_folder, get_columns
from models.utils import get_biggest_change_features, load_model, neutralize, save_model
import pandas as pd
import numerapi
import warnings
warnings.filterwarnings("ignore")
import json
from lightgbm import LGBMRegressor
import json
import logging


logger = logging.getLogger("LGBM")

@dataclass
class LGBM:
    numerai_public_key: Optional[str] = None
    numerai_secret_key: Optional[str] = None
    numerai_model_name: Optional[str] = None
    
    DATA_FOLDER: str = "data"
    features_set: Literal["full", "small"] = "full"
    PREDICTION_COL: str = "prediction"
    TARGET_COL: str = 'target'
    ERA_COL: str = 'era'
    DATA_TYPE_COL: str = 'data_type'
    EXAMPLE_PREDS_COL: str = "example_preds"
    MODEL_NAME: str = "lbgm"
    
    
    def __post_init__(self):        
        logger.info("Connecting to Numerai server")
        self.numerapi = numerapi.NumerAPI(self.numerai_public_key, self.numerai_secret_key)
        logger.info("Successfully logged into Numerai API")
        self.current_round = self.numerapi.get_current_round()
        logger.info(f"Current Round : {self.current_round}")
        
        self.ROUND_FOLDER = os.path.join(self.DATA_FOLDER, str(self.current_round))
        self.INPUT_FOLDER = os.path.join(self.ROUND_FOLDER, "input")
        self.OUTPUT_FOLDER = os.path.join(self.ROUND_FOLDER, "output", "lgbm", self.features_set)
        self.MODEL_FOLDER = os.path.join(self.ROUND_FOLDER, 'cache', self.features_set)
        
        create_folder(self.DATA_FOLDER)
        create_folder(self.ROUND_FOLDER)
        create_folder(self.INPUT_FOLDER)
        create_folder(self.OUTPUT_FOLDER)
        create_folder(self.MODEL_FOLDER)
        
        self.TRAINING_DATA_FILE = os.path.join(self.INPUT_FOLDER, "training_data.parquet")
        self.TOURNAMENT_DATA_FILE = os.path.join(self.INPUT_FOLDER, f"tournament_data.parquet")
        self.VALIDATION_DATA_FILE = os.path.join(self.INPUT_FOLDER, "validation_data.parquet")
        self.EXAMPLE_VALIDATION_PREDICTIONS_FILE = os.path.join(self.INPUT_FOLDER, "example_validation_predictions.parquet")
        self.FEATURES_FILE = os.path.join(self.INPUT_FOLDER, "features.json")
        self.TARGET_MODEL_FILE = os.path.join(self.MODEL_FOLDER, f"{self.MODEL_NAME}.model")
        self.FEATURES_CORR_FILE = os.path.join(self.MODEL_FOLDER, "features_corr.csv")
        self.VALIDATION_PREDICTIONS_FILE = os.path.join(self.OUTPUT_FOLDER, f"validation_predictions.csv")
        self.TOURNAMENT_PREDICTIONS_FILE = os.path.join(self.OUTPUT_FOLDER, f"tournament_predictions.csv")
        
    
    def download_dataset(self):
        # Tournament data changes every week so we specify the round in their name. Training
        # and validation data only change periodically, so no need to download them every time.
        logger.info('Downloading dataset files...')
        self.numerapi.download_dataset("numerai_training_data.parquet", self.TRAINING_DATA_FILE)
        self.numerapi.download_dataset("numerai_tournament_data.parquet", self.TOURNAMENT_DATA_FILE)
        self.numerapi.download_dataset("numerai_validation_data.parquet", self.VALIDATION_DATA_FILE)
        self.numerapi.download_dataset("example_validation_predictions.parquet", self.EXAMPLE_VALIDATION_PREDICTIONS_FILE)
        self.numerapi.download_dataset("features.json", self.FEATURES_FILE)
        logger.info("All files ready")
    
    
    def load_training_data(self):
        all_columns, features, targets, other_cols = get_columns(self.TRAINING_DATA_FILE)

        logger.info('Reading minimal training data')
        # read the feature metadata and get the "small" feature set
        
        if self.features_set != 'full' and self.FEATURES_FILE is not None:
            with open(self.FEATURES_FILE, "r") as f:
                feature_metadata = json.load(f)
            features = feature_metadata["feature_sets"][self.features_set]
        # read in just those features along with era and target columns
        read_columns = features + targets + other_cols
        # note: sometimes when trying to read the downloaded data you get an error about invalid magic parquet bytes...
        # if so, delete the file and rerun the napi.download_dataset to fix the corrupted file
        self.training_data = pd.read_parquet(self.TRAINING_DATA_FILE, columns=read_columns)
        self.all_columns = all_columns
        self.features = features
        self.targets = targets
        self.other_cols = other_cols


    def train_model(self):
        filename = self.TARGET_MODEL_FILE
        if os.path.exists(filename):
            self.model = load_model(filename)
        else:
            params = {"n_estimators": 2000,
                        "learning_rate": 0.01,
                        "max_depth": 5,
                        "num_leaves": 2 ** 5,
                        "colsample_bytree": 0.1}
            model = LGBMRegressor(**params)

            # train on all of train and save the model so we don't have to train next time
            model.fit(self.training_data.filter(like='feature_', axis='columns'), self.training_data[self.TARGET_COL])
            logger.info(f"saving new model: {filename}")
            save_model(model, filename)
            self.model = model

    
    def load_tournament_data(self):
        self.tournament_data = pd.read_parquet(self.TOURNAMENT_DATA_FILE, columns=self.all_columns)
        tournament_data_features_only = self.tournament_data[self.features + [self.DATA_TYPE_COL]]
        nans_per_col = tournament_data_features_only[tournament_data_features_only["data_type"] == "live"].isna().sum()
        del tournament_data_features_only
        # check for nans and fill nans
        if nans_per_col.any():
            total_rows = len(self.tournament_data[self.tournament_data[self.DATA_TYPE_COL] == "live"])
            logger.info(f"Number of nans per column this week: {nans_per_col[nans_per_col > 0]}")
            logger.info(f"out of {total_rows} total rows")
            logger.info(f"filling nans with 0.5")
            self.tournament_data.loc[:, self.features] = self.tournament_data.loc[:, self.features].fillna(0.5)
        else:
            logger.info("No nans in the features this week!")
    
    
    def check_model_features(self):
        # double check the feature that the model expects vs what is available to prevent our
        # pipeline from failing if Numerai adds more data and we don't have time to retrain!
        self.model_expected_features = self.model.booster_.feature_name()
        if set(self.model_expected_features) != set(self.features):
            logger.warn(f"New features are available! Might want to retrain model {self.MODEL_NAME}.")
    
    
    def calculate_riskiest_features(self, neutralize_riskiest: int = 168):
        if not os.path.exists(self.FEATURES_CORR_FILE):
            # getting the per era correlation of each feature vs the target
            all_feature_corrs = self.training_data.groupby(self.ERA_COL).apply(
                lambda era: era[self.features].corrwith(era[self.TARGET_COL])
            )
            all_feature_corrs.to_csv(self.FEATURES_CORR_FILE)
        else:
            all_feature_corrs = pd.read_csv(self.FEATURES_CORR_FILE).set_index('era')
        
        self.riskiest_features = get_biggest_change_features(all_feature_corrs, neutralize_riskiest)
    
    
    def upload_predictions(self) -> None:
        model_uuid = self.numerapi.get_models().get(self.numerai_model_name)
        logger.info(f"Uploading predictions to Numerai API for model : {self.numerai_model_name} (uuid: {model_uuid})")
        if self.numerapi.submission_status(model_uuid) is None:
            self.numerapi.upload_predictions(self.TOURNAMENT_PREDICTIONS_FILE, model_id=model_uuid)
            logger.info("Predictions uploaded")
        else:
            logger.info("Predictions already uploaded, skipping...")
    
    def run(self, neutralize_riskiest: int = 168, upload_predictions: bool=False, generate_validation_file: bool=False) -> None:
        self.download_dataset()
        self.load_training_data()
        self.train_model()
        self.load_tournament_data()
        self.check_model_features()
        self.calculate_riskiest_features(neutralize_riskiest)
        
        if generate_validation_file:
            self.validation_data = pd.read_parquet(self.VALIDATION_DATA_FILE, columns=self.all_columns)
            self.validation_data[self.PREDICTION_COL] = self.model.predict(self.validation_data[self.model_expected_features])
            # neutralize our predictions to the riskiest features
            validation_result = neutralize(
                df=self.validation_data,
                columns=[self.PREDICTION_COL],
                neutralizers=self.riskiest_features,
                proportion=1.0,
                normalize=True,
                era_col=self.ERA_COL
            )
            validation_result = validation_result.rank(pct=True)
            validation_result.to_csv(self.VALIDATION_PREDICTIONS_FILE)

        
        self.tournament_data[self.PREDICTION_COL] = self.model.predict(self.tournament_data[self.model_expected_features])
        tournament_result = neutralize(
            df=self.tournament_data,
            columns=[self.PREDICTION_COL],
            neutralizers=self.riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=self.ERA_COL
        )
        
        tournament_result = tournament_result.rank(pct=True)
        tournament_result.to_csv(self.TOURNAMENT_PREDICTIONS_FILE)
        
        if upload_predictions:
            self.upload_predictions()

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
    LGBM().run(upload_predictions=True)
    # values = [160 + x for x in range(10)]
    # print(values)
    # run(neutralize_riskiest=values)
    # run(neutralize_riskiest=50, features_set='small')
    print((datetime.now() - start).total_seconds())
