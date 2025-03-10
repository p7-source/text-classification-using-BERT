from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    training_data_path: Path
    training_data_file: str
    training_cleansed_data: str