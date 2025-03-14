from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments, 
    DataCollatorWithPadding,
    AutoTokenizer
)
from pathlib import Path
import torch
from textClassifier import logger
from textClassifier.entity.config_entity import TrainingConfig


class ModelTraining:
    def __init__(self, config: TrainingConfig):
        """
        Initializes the ModelTraining class.

        Args:
            config (TrainingConfig): Configuration for model training.
        """
        self.config = config

    def load_datasets(self):
        """
        Loads the train, validation, and test datasets from the specified directory.
        """
        # Explicitly set the correct path
        datasets = Path("artifacts/feature_engineering/datasets")
        
        # Debug: Print the datasets directory
        print("Datasets Directory:", datasets)

        # Load datasets
        train_dataset = torch.load(datasets / "train_dataset.pt", weights_only=False)
        val_dataset = torch.load(datasets / "val_dataset.pt", weights_only=False)
        test_dataset = torch.load(datasets / "test_dataset.pt", weights_only=False)

        logger.info(f"Datasets loaded from {datasets}")
        return train_dataset, val_dataset, test_dataset

    def train(self):
        """
        Trains the model using the loaded datasets.
        """
        # Load datasets
        train_dataset, val_dataset, _ = self.load_datasets()

        # Load the base model and tokenizer from the artifacts/prepare_model folder
        # base_model_path = Path(self.config.base_model_path)
        base_model_path = Path("artifacts/prepare_model")
        model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            max_steps=self.config.max_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="epoch",
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        # Start training
        trainer.train()

        # Save the trained model
        self.save_model(trainer)

    def save_model(self, trainer):
        """
        Saves the trained model and tokenizer to the specified directory.

        Args:
            trainer (Trainer): The Trainer object containing the trained model.
        """
        save_path = Path(self.config.model_save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the model
        trainer.save_model(save_path)
        logger.info(f"Model saved to {save_path}")

        # Save the tokenizer
        tokenizer = trainer.tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
            logger.info(f"Tokenizer saved to {save_path}")
        else:
            logger.warning("Tokenizer not found in the trainer object. Only the model was saved.")

    def evaluate(self):
        """
        Evaluates the model on the test dataset.
        """
        # Load the test dataset
        _, _, test_dataset = self.load_datasets()

        # Load the trained model from the model_save_path
        model = AutoModelForSequenceClassification.from_pretrained(self.config.model_save_path)

        # Set up training arguments for evaluation
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size
        )

        # Initialize the Trainer for evaluation
        trainer = Trainer(
            model=model,
            args=training_args
        )

        # Evaluate the model
        results = trainer.evaluate(test_dataset)

        logger.info("Evaluation Results:")
        logger.info(f"  - Loss: {results['eval_loss']:.4f}")
        logger.info(f"  - Runtime: {results['eval_runtime']:.2f} seconds")
        logger.info(f"  - Samples per Second: {results['eval_samples_per_second']:.2f}")
        logger.info(f"  - Steps per Second: {results['eval_steps_per_second']:.2f}")
        logger.info(f"  - Epoch: {results.get('epoch', 'N/A')}")