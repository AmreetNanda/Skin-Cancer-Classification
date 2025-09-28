from src.skin_disease_classifier.components.prepare_base_model import build_cnn
from src.skin_disease_classifier.config.configuration import ConfigurationManager

if __name__ == "__main__":
    config = ConfigurationManager().get_base_model_config()
    model = build_cnn(input_shape=config.input_shape, num_classes=config.num_classes)
    model.summary()
    model.save(config.save_path)
    print(f"✅ Base model saved at {config.save_path}")
