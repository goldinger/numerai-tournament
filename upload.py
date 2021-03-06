import os
from models.lgbm import LGBM
import logging
from models.utils import create_folder


create_folder('logs')
logging.basicConfig(filename='logs/upload.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(name)s: %(message)s')

def run_lgbm() -> None:
    model = LGBM(
        numerai_model_name=os.environ.get('NUMERAI_LGBM_MODEL_NAME'),
        numerai_public_key=os.environ.get('NUMERAI_PUBLIC_KEY'),
        numerai_secret_key=os.environ.get('NUMERAI_SECRET_KEY'),
        DATA_FOLDER=os.environ.get("DATA_DIR", "data")
        )
    model.run(upload_predictions=True)


def main() -> None:
    run_lgbm()

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    run_lgbm()
