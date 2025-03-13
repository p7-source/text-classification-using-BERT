from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """
        Initializes and saves the base model.
        """
        # Initialize the model
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3
        )
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Save the model
        self.save_model(model, tokenizer)
        # self.save_model(tokenizer)

    def save_model(self, model, tokenizer):
        """
        Saves the model to the specified path.
        """
        save_path = self.config.base_model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"Base model saved to {save_path}")
        print(f"Tokenizer saved to {save_path}")