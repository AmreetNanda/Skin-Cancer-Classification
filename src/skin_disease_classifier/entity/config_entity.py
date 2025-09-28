from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    metadata_csv: str
    image_dir: str
    output_csv: str

@dataclass
class BaseModelConfig:
    input_shape: tuple
    num_classes: int
    save_path: str

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    report_dir: str
    model_dir: str
