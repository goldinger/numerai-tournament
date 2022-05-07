from models.lgbm import LGBM


if __name__ == '__main__':
    LGBM().run(upload_predictions=True)