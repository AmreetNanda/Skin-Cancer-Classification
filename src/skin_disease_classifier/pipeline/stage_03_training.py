import pandas as pd
from src.skin_disease_classifier.components.prepare_base_model import build_cnn
from src.skin_disease_classifier.components.prepare_callbacks import get_callbacks
from src.skin_disease_classifier.components.training import Trainer
from src.skin_disease_classifier.config.configuration import ConfigurationManager

if __name__ == "__main__":
    cfg_manager = ConfigurationManager()
    data_cfg = cfg_manager.get_data_ingestion_config()
    base_cfg = cfg_manager.get_base_model_config()
    train_cfg = cfg_manager.get_training_config()

    df = pd.read_csv(data_cfg.output_csv)

    # Load prebuilt model
    model = build_cnn(input_shape=base_cfg.input_shape, num_classes=base_cfg.num_classes)
    callbacks = get_callbacks(train_cfg.model_dir)

    trainer = Trainer(model, callbacks, train_cfg)
    model, test_data = trainer.train(df)

    model.save(f"{train_cfg.model_dir}/final_model.h5")
    print("✅ Training complete. Final model saved.")
