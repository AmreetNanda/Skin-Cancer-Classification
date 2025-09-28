import subprocess

if __name__ == "__main__":
    print("🚀 Stage 01: Data Ingestion")
    subprocess.run(["python", "src/skin_disease_classifier/skin_classification/pipeline/stage_01_data_ingestion.py"])

    print("🚀 Stage 02: Prepare Base Model")
    subprocess.run(["python", "src/skin_disease_classifier/skin_classification/pipeline/stage_02_prepare_base_model.py"])

    print("🚀 Stage 03: Training")
    subprocess.run(["python", "src/skin_disease_classifier/skin_classification/pipeline/stage_03_training.py"])

    print("🚀 Stage 04: Evaluation")
    subprocess.run(["python", "src/skin_disease_classifier/skin_classification/pipeline/stage_04_evaluation.py"])

    print("✅ All pipeline stages executed.")
