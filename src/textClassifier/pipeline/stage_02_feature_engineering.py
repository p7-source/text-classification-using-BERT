from textClassifier.config.configuration import ConfigurationManager
from textClassifier.components.feature_engineering import DataPreprocessing
from textClassifier import logger

STAGE_NAME = "Feature Engineering Stage"

class FeatureEngineeringPipeline:
    def __init__(self):
        pass

    def main(self):
        # Example usage

        # Assuming ConfigurationManager is defined elsewhere
        config_manager = ConfigurationManager()

        # Get the data preprocessing config
        get_data_pre_config = config_manager.get_data_preprocessing_config()

        # Initialize DataPreprocessing with the config
        data_processing = DataPreprocessing(config=get_data_pre_config)

        # Load and preprocess the data
        output = data_processing.get_data_from_features()

        # Save the preprocessed data
        data_processing.save_data(data=output, path_name='cleaned_tweets.csv')

        # Map labels to numerical values
        output = data_processing.mapping_labels_func(data=output)

        # Tokenize the cleaned text
        output = data_processing.tokenize_text(data=output)

        # Split the data into train, validation, and test sets
        splits = data_processing.train_val_test_split(data=output)

        # Access the splits
        train_data = splits['train']
        val_data = splits['val']
        test_data = splits['test']

        # # Display the sizes of the splits
        # print(f"Train set size: {len(train_data['texts'])}")
        # print(f"Validation set size: {len(val_data['texts'])}")
        # print(f"Test set size: {len(test_data['texts'])}")



