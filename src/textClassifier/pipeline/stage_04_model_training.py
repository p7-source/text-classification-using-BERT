from textClassifier.config.configuration import ConfigurationManager
from textClassifier.components.model_training import ModelTraining
from textClassifier import logger

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        trainingConfig = config.get_training_config()        
        model_training = ModelTraining(config=trainingConfig)
        model_training.train()
        model_training.evaluate()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e