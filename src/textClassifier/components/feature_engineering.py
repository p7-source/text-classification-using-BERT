from textClassifier import logger
import zipfile
import os
from textClassifier.utils.commons import get_size
import gdown
import spacy
import string
import pandas as pd
from pathlib import Path
from textClassifier.entity.config_entity import DataPreprocessingConfig
from textClassifier import logger
import pandas as pd
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split





######################


import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import spacy
from nltk.corpus import stopwords
import string
import logging

logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        """
        Custom PyTorch Dataset for sentiment analysis.

        Args:
            encodings (dict): Tokenized encodings (e.g., input_ids, attention_mask).
            labels (list): List of labels corresponding to the encodings.
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """
        Initializes the DataPreprocessing class.

        Args:
            config: Configuration object containing paths and settings.
        """
        self.config = config
        self.nlp = spacy.load('en_core_web_sm')  # Load spaCy model
        self.stop_words = set(stopwords.words('english'))  # Load stopwords
        self.mapping_labels = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        
        # Load the dataset during initialization
        self.data = self._load_data()

        # Set the default output file path
        self.output_file_path = Path(self.config.training_cleansed_data) / 'cleaned_tweets.csv'
        self.datasets_dir = Path(self.config.datasets_dir)

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from the path specified in the config.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        # Check if the file exists
        if not Path(self.config.training_data_file).exists():
            raise FileNotFoundError(f"File not found: {self.config.training_data_file}")

        # Load the dataset
        data = pd.read_csv(self.config.training_data_file)
        logger.info(f"Dataset loaded from {self.config.training_data_file}")
        return data

    def text_process(self) -> None:
        """
        Preprocesses the text column in the dataset by:
        1. Converting to lowercase
        2. Removing punctuation
        3. Tokenizing and lemmatizing
        4. Removing stopwords
        """
        # Apply preprocessing to the text column
        self.data['cleaned_text'] = self.data['text'].apply(self._process_single_text)
        logger.info("Text preprocessing completed.")

    def _process_single_text(self, text: str) -> str:
        """
        Helper method to preprocess a single text string.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize and lemmatize
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in self.stop_words]

        return ' '.join(tokens)

    def mapping_labels_func(self) -> None:
        """
        Maps the sentiment labels to numerical values.
        """
        self.data['label'] = self.data['airline_sentiment'].map(self.mapping_labels)
        logger.info("Labels mapped to numerical values.")

    def tokenize_text(self) -> None:
        """
        Tokenizes the cleaned text column using BERT tokenizer.
        """
        # Tokenize the text
        self.data['tokenized'] = self.data['cleaned_text'].apply(
            lambda x: self.tokenizer(x, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        )
        logger.info("Text tokenization completed.")

    def save_data(self) -> None:
        """
        Saves the preprocessed data to the default output file.
        """
        # Save the DataFrame to CSV
        self.data.to_csv(self.output_file_path, index=False)
        logger.info(f"Preprocessed data saved to {self.output_file_path}")

    def train_val_test_split(self, test_size: float = 0.3, val_size: float = 0.5, random_state: int = 42) -> dict:
        """
        Splits the dataset into training, validation, and test sets.

        Args:
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
            self.data['cleaned_text'].to_list(), self.data['label'].to_list(), test_size=test_size, random_state=random_state
        )

        # Split temp into validation and test
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=val_size, random_state=random_state
        )

        logger.info("Data split into train, validation, and test sets.")
        return {
            'train': {'texts': train_texts, 'labels': train_labels},
            'val': {'texts': val_texts, 'labels': val_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }

    def convert_to_tokenized_datasets(self, splits: dict) -> dict:
        """
        Converts the train, validation, and test splits into PyTorch datasets.

        Args:
            splits (dict): A dictionary containing the splits:
                {
                    'train': {'texts': train_texts, 'labels': train_labels},
                    'val': {'texts': val_texts, 'labels': val_labels},
                    'test': {'texts': test_texts, 'labels': test_labels}
                }

        Returns:
            dict: A dictionary containing the PyTorch datasets:
                {
                    'train': train_dataset,
                    'val': val_dataset,
                    'test': test_dataset
                }
        """
        # Tokenize the texts
        train_encodings = self.tokenizer(splits['train']['texts'], truncation=True, padding=True, max_length=128)
        val_encodings = self.tokenizer(splits['val']['texts'], truncation=True, padding=True, max_length=128)
        test_encodings = self.tokenizer(splits['test']['texts'], truncation=True, padding=True, max_length=128)

        # Create PyTorch datasets
        train_dataset = SentimentDataset(train_encodings, splits['train']['labels'])
        val_dataset = SentimentDataset(val_encodings, splits['val']['labels'])
        test_dataset = SentimentDataset(test_encodings, splits['test']['labels'])

        logger.info("PyTorch datasets created successfully.")
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    def save_datasets(self, train_dataset, val_dataset, test_dataset):
        """
        Saves the train, validation, and test datasets to the specified directory.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            test_dataset: Test dataset.
        """
        # Create the directory if it doesn't exist
        datasets_dir = Path(self.config.datasets_dir)
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # Save the datasets
        torch.save(train_dataset, datasets_dir / "train_dataset.pt")
        torch.save(val_dataset, datasets_dir / "val_dataset.pt")
        torch.save(test_dataset, datasets_dir / "test_dataset.pt")

        logger.info(f"Datasets saved to {datasets_dir}")

# class DataPreprocessing:
#     def __init__(self, config: DataPreprocessingConfig):
#         self.config = config
#         self.nlp = spacy.load('en_core_web_sm')  # Load spaCy model
#         self.stop_words = set(stopwords.words('english'))  # Load stopwords
#         self.mapping_labels = {'negative': 0, 'neutral': 1, 'positive': 2}
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     def text_process(self, text: str) -> str:
#         """
#         Preprocesses the input text by:
#         1. Converting to lowercase
#         2. Removing punctuation
#         3. Tokenizing and lemmatizing
#         4. Removing stopwords
#         """
#         # Convert to lowercase
#         text = text.lower()

#         # Remove punctuation
#         text = text.translate(str.maketrans('', '', string.punctuation))

#         # Tokenize and lemmatize
#         doc = self.nlp(text)
#         tokens = [token.lemma_ for token in doc if token.text not in self.stop_words]

#         return ' '.join(tokens)

#     def get_data_from_features(self) -> pd.DataFrame:
#         """
#         Loads the dataset from the path specified in the config and applies preprocessing to the text column.
#         """
#         # Check if the file exists
#         if not Path(self.config.training_data_file).exists():
#             raise FileNotFoundError(f"File not found: {self.config.training_data_file}")

#         # Load the dataset
#         data = pd.read_csv(self.config.training_data_file)

#         # Apply preprocessing to the text column
#         data['cleaned_text'] = data['text'].apply(self.text_process)

#         return data

#     def save_data(self, data: pd.DataFrame, path_name: str = 'cleaned_tweets.csv') -> None:
#         """
#         Saves the preprocessed data to the specified output file.

#         Args:
#             data (pd.DataFrame): The preprocessed DataFrame to save.
#             path_name (str): The name of the output file. Defaults to "cleaned_tweets.csv".
#         """
#         # Create the full output file path
#         out_path_name = Path(self.config.training_cleansed_data) / path_name

#         # Save the DataFrame to CSV
#         data.to_csv(out_path_name, index=False)
#         logger.info(f"Preprocessed data saved to {out_path_name}")


#     def mapping_labels_func(self, data: pd.DataFrame) -> pd.DataFrame:

#         data['label'] = data['airline_sentiment'].map(self.mapping_labels)
#         logger.info(f"values are mapped in the {data}")
#         return data
    
#     # def tokenizing_func(self, data: pd.DataFrame, text_column: str = 'cleaned_text') -> dict:
#     #     # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     #     # return tokenizer(data, padding='max_length', truncation=True, max_length=128)
    
#     #     tokenized_inputs = self.tokenizer(
#     #         data[text_column].tolist(),  # Convert text column to list
#     #         padding='max_length',        # Pad to max length
#     #         truncation=True,             # Truncate to max length
#     #         max_length=128,              # Set max length
#     #         return_tensors='pt'          # Return PyTorch tensors
#     #     )
#     #     return tokenized_inputs

#     def tokenize_text(self, data: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
#         """
#         Tokenizes the text in the specified column using BERT tokenizer.

#         Args:
#             data (pd.DataFrame): The DataFrame containing the text to tokenize.
#             text_column (str): The name of the column containing the text. Defaults to 'cleaned_text'.

#         Returns:
#             pd.DataFrame: The DataFrame with an additional 'tokenized' column.
#         """
#         # Tokenize the text
#         data['tokenized'] = data[text_column].apply(
#             lambda x: self.tokenizer(x, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
#         )
#         logger.info(f"Text in column '{text_column}' has been tokenized.")
#         return data
    
#     def train_val_test_split(self, data: pd.DataFrame, test_size: float = 0.3, val_size: float = 0.5, random_state: int = 42):
#         """
#         Splits the dataset into training, validation, and test sets.

#         Args:
#             data (pd.DataFrame): The DataFrame containing the tokenized data and labels.
#             test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.3.
#             val_size (float): Proportion of the temp split to include in the validation split. Defaults to 0.5.
#             random_state (int): Random seed for reproducibility. Defaults to 42.

#         Returns:
#             A dictionary containing the splits:
#             {
#                 'train': {'texts': train_texts, 'labels': train_labels},
#                 'val': {'texts': val_texts, 'labels': val_labels},
#                 'test': {'texts': test_texts, 'labels': test_labels}
#             }
#         """
#         # Split into train and temp (val + test)
#         train_texts, temp_texts, train_labels, temp_labels = train_test_split(
#             data['tokenized'].to_list(), data['label'].to_list(), test_size=test_size, random_state=random_state
#         )

#         # Split temp into validation and test
#         val_texts, test_texts, val_labels, test_labels = train_test_split(
#             temp_texts, temp_labels, test_size=val_size, random_state=random_state
#         )

#         logger.info(f"Data split into train, validation, and test sets.")
#         return {
#             'train': {'texts': train_texts, 'labels': train_labels},
#             'val': {'texts': val_texts, 'labels': val_labels},
#             'test': {'texts': test_texts, 'labels': test_labels}
#         }

