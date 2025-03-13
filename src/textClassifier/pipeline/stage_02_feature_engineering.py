from textClassifier.config.configuration import ConfigurationManager
from textClassifier.components.feature_engineering import DataPreprocessing
from textClassifier import logger

STAGE_NAME = "Feature Engineering Stage"

class FeatureEngineeringPipeline:
    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()
        get_data_pre_config = config.get_data_preprocessing_config()
        data_pre_process = DataPreprocessing(config=get_data_pre_config)
    
    # Preprocess the data
        data_pre_process.text_process()
        data_pre_process.mapping_labels_func()
        data_pre_process.tokenize_text()
    
    # Save the preprocessed data
        data_pre_process.save_data()  # No need to pass the path
    
    # Split the data into train, validation, and test sets
        splits = data_pre_process.train_val_test_split()
    
    # Convert splits to PyTorch datasets
        tokenized_datasets = data_pre_process.convert_to_tokenized_datasets(splits)
    
    # Access the PyTorch datasets
        train_dataset = tokenized_datasets['train']
        val_dataset = tokenized_datasets['val']
        test_dataset = tokenized_datasets['test']
        logger.info("PyTorch datasets created successfully.")
        datasets = data_pre_process.save_datasets(train_dataset, val_dataset, test_dataset)

    

        # # Example usage

        # # Assuming ConfigurationManager is defined elsewhere
        # config_manager = ConfigurationManager()

        # # Get the data preprocessing config
        # get_data_pre_config = config_manager.get_data_preprocessing_config()

        # # Initialize DataPreprocessing with the config
        # data_processing = DataPreprocessing(config=get_data_pre_config)

        # # Load and preprocess the data
        # output = data_processing.get_data_from_features()

        # # Save the preprocessed data
        # data_processing.save_data(data=output, path_name='cleaned_tweets.csv')

        # # Map labels to numerical values
        # output = data_processing.mapping_labels_func(data=output)

        # # Tokenize the cleaned text
        # output = data_processing.tokenize_text(data=output)

        # # Split the data into train, validation, and test sets
        # splits = data_processing.train_val_test_split(data=output)

        # # Access the splits
        # train_data = splits['train']
        # val_data = splits['val']
        # test_data = splits['test']

        # # # Display the sizes of the splits
        # # print(f"Train set size: {len(train_data['texts'])}")
        # # print(f"Validation set size: {len(val_data['texts'])}")
        # # print(f"Test set size: {len(test_data['texts'])}")



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureEngineeringPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e