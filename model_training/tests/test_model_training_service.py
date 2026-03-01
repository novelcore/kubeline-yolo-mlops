# This file is intentionally empty.
#
# The model training service tests live in test_training_service.py, which
# covers TrainingService (the real implementation) with all heavy dependencies
# (ultralytics, mlflow, boto3) mocked.
#
# The old ModelTrainingService stub has been replaced by TrainingService in
# app/services/model_training.py as part of the YOLO + MLflow implementation.
