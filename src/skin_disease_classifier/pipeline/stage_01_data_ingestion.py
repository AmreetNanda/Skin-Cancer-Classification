from src.skin_disease_classifier.components.data_ingestion import DataIngestion
from src.skin_disease_classifier.config.configuration import ConfigurationManager

if __name__ == "__main__":
    config = ConfigurationManager().get_data_ingestion_config()
    ingestion = DataIngestion(config)
    df = ingestion.load_metadata()
    df = ingestion.map_image_paths(df)
    df = ingestion.enrich_metadata(df)
    balanced = ingestion.balance_dataset(df)
    ingestion.save(balanced, config.output_csv)
    print("✅ Data ingestion complete. Balanced dataset saved.")
