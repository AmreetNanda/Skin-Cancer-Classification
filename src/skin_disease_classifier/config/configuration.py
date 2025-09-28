import yaml
import os
from src.skin_disease_classifier.entity.config_entity import (
    DataIngestionConfig, BaseModelConfig, TrainingConfig
)

class ConfigurationManager:
    def __init__(self, config_path="Config/config.yaml", params_path="Config/params.yaml"):
        self.config = self.read_yaml(config_path)
        self.params = self.read_yaml(params_path)
        self.artifacts_root = self.config["artifacts_root"]
        os.makedirs(self.artifacts_root, exist_ok=True)

    @staticmethod
    def read_yaml(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        return DataIngestionConfig(
            metadata_csv=cfg["metadata_csv"],
            image_dir=cfg["image_dir"],
            output_csv=cfg["output_csv"],
        )

    def get_base_model_config(self) -> BaseModelConfig:
        cfg = self.config["base_model"]
        return BaseModelConfig(
            input_shape=tuple(cfg["input_shape"]),
            num_classes=cfg["num_classes"],
            save_path=cfg["save_path"],
        )

    def get_training_config(self) -> TrainingConfig:
        cfg = self.config["training"]
        return TrainingConfig(
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            report_dir=cfg["report_dir"],
            model_dir=cfg["model_dir"],
        )
