from textClassifier import logger
from textClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textClassifier.pipeline.stage_02_feature_engineering import FeatureEngineeringPipeline
from textClassifier.pipeline.stage_03_base_model import BaseModelPipeline

STAGE_NAME = "Data Ingestion Stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Feature Engineering Stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = FeatureEngineeringPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n x==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Preparing Base Model Stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = BaseModelPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n x==========x")
except Exception as e:
    logger.exception(e)
    raise e