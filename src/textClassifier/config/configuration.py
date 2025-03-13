from textClassifier.constants import *
from textClassifier.utils.commons import read_yaml, create_directories
from textClassifier.entity.config_entity import (DataIngestionConfig,
                                                 DataPreprocessingConfig,
                                                 PrepareBaseModelConfig
                                                 )


class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.feature_engineering

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            training_data_path=config.training_data_path,
            training_data_file=config.training_data_file,
            training_cleansed_data=config.training_cleansed_data,
            datasets_dir=config.datasets_dir
        )

        return data_preprocessing_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            params_num_train_epochs=self.params.num_train_epochs,
            params_per_device_train_batch_size=self.params.per_device_train_batch_size,
            params_per_device_eval_batch_size=self.params.per_device_eval_batch_size,
            params_weight_decay=self.params.weight_decay,
            params_max_steps=self.params.max_steps,
            params_warmup_steps = self.params.warmup_steps,
            params_save_steps=self.params.save_steps,
            params_logging_steps=self.params.logging_steps



        )

        return prepare_base_model_config