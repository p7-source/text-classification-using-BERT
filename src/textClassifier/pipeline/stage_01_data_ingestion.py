from textClassifier.config.configuration import ConfigurationManager
from textClassifier.components.data_ingestion import DataIngestion
from textClassifier import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        dataIngestionConfig = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=dataIngestionConfig)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e