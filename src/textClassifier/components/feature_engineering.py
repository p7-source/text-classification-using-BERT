from textClassifier import logger
import zipfile
import os
from textClassifier.utils.commons import get_size
import gdown
import spacy
import string
import pandas as pd
from pathlib import Path
from textClassifier.entity import DataPreprocessingConfig
from textClassifier import logger
import pandas as pd
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split



class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.nlp = spacy.load('en_core_web_sm')  # Load spaCy model
        self.stop_words = set(stopwords.words('english'))  # Load stopwords
        self.mapping_labels = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def text_process(self, text: str) -> str:
        """
        Preprocesses the input text by:
        1. Converting to lowercase
        2. Removing punctuation
        3. Tokenizing and lemmatizing
        4. Removing stopwords
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize and lemmatize
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in self.stop_words]

        return ' '.join(tokens)

    def get_data_from_features(self) -> pd.DataFrame:
        """
        Loads the dataset from the path specified in the config and applies preprocessing to the text column.
        """
        # Check if the file exists
        if not Path(self.config.training_data_file).exists():
            raise FileNotFoundError(f"File not found: {self.config.training_data_file}")

        # Load the dataset
        data = pd.read_csv(self.config.training_data_file)

        # Apply preprocessing to the text column
        data['cleaned_text'] = data['text'].apply(self.text_process)

        return data

    def save_data(self, data: pd.DataFrame, path_name: str = 'cleaned_tweets.csv') -> None:
        """
        Saves the preprocessed data to the specified output file.

        Args:
            data (pd.DataFrame): The preprocessed DataFrame to save.
            path_name (str): The name of the output file. Defaults to "cleaned_tweets.csv".
        """
        # Create the full output file path
        out_path_name = Path(self.config.training_cleansed_data) / path_name

        # Save the DataFrame to CSV
        data.to_csv(out_path_name, index=False)
        logger.info(f"Preprocessed data saved to {out_path_name}")


    def mapping_labels_func(self, data: pd.DataFrame) -> pd.DataFrame:

        data['label'] = data['airline_sentiment'].map(self.mapping_labels)
        logger.info(f"values are mapped in the {data}")
        return data
    
    # def tokenizing_func(self, data: pd.DataFrame, text_column: str = 'cleaned_text') -> dict:
    #     # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     # return tokenizer(data, padding='max_length', truncation=True, max_length=128)
    
    #     tokenized_inputs = self.tokenizer(
    #         data[text_column].tolist(),  # Convert text column to list
    #         padding='max_length',        # Pad to max length
    #         truncation=True,             # Truncate to max length
    #         max_length=128,              # Set max length
    #         return_tensors='pt'          # Return PyTorch tensors
    #     )
    #     return tokenized_inputs

    def tokenize_text(self, data: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Tokenizes the text in the specified column using BERT tokenizer.

        Args:
            data (pd.DataFrame): The DataFrame containing the text to tokenize.
            text_column (str): The name of the column containing the text. Defaults to 'cleaned_text'.

        Returns:
            pd.DataFrame: The DataFrame with an additional 'tokenized' column.
        """
        # Tokenize the text
        data['tokenized'] = data[text_column].apply(
            lambda x: self.tokenizer(x, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        )
        logger.info(f"Text in column '{text_column}' has been tokenized.")
        return data
    
    def train_val_test_split(self, data: pd.DataFrame, test_size: float = 0.3, val_size: float = 0.5, random_state: int = 42):
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            data (pd.DataFrame): The DataFrame containing the tokenized data and labels.
            test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.3.
            val_size (float): Proportion of the temp split to include in the validation split. Defaults to 0.5.
            random_state (int): Random seed for reproducibility. Defaults to 42.

        Returns:
            A dictionary containing the splits:
            {
                'train': {'texts': train_texts, 'labels': train_labels},
                'val': {'texts': val_texts, 'labels': val_labels},
                'test': {'texts': test_texts, 'labels': test_labels}
            }
        """
        # Split into train and temp (val + test)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            data['tokenized'].to_list(), data['label'].to_list(), test_size=test_size, random_state=random_state
        )

        # Split temp into validation and test
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=val_size, random_state=random_state
        )

        logger.info(f"Data split into train, validation, and test sets.")
        return {
            'train': {'texts': train_texts, 'labels': train_labels},
            'val': {'texts': val_texts, 'labels': val_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }

