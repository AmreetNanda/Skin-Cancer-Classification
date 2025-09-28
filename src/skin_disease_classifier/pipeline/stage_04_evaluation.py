import pandas as pd
from src.skin_disease_classifier.components.evaluation import Evaluator
from src.skin_disease_classifier.config.configuration import ConfigurationManager

if __name__ == "__main__":
    cfg_manager = ConfigurationManager()
    data_cfg = cfg_manager.get_data_ingestion_config()
    train_cfg = cfg_manager.get_training_config()

    df = pd.read_csv(data_cfg.output_csv)
    categories = df['cell_type'].unique().tolist()

    evaluator = Evaluator(
        model_path=f"{train_cfg.model_dir}/final_model.h5",
        report_dir=train_cfg.report_dir
    )

    # NOTE: You need to reload x_test, y_test from training stage or persist them
    # Here we just print instructions
    print("⚠️ Load x_test, y_test from artifacts before calling evaluate()")
