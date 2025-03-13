from textClassifier.config.configuration import ConfigurationManager
from textClassifier.components.base_model import PrepareBaseModel
from textClassifier import logger

STAGE_NAME = "Preparing Base Model"

class BaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        baseModelConfig = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=baseModelConfig)
        prepare_base_model.get_base_model()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BaseModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e