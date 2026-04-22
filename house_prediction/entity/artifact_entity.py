from dataclasses import dataclass

@dataclass
# Information returned at the end of a pipeline(ingestion,validation,transformation,modeltrainer)

class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str

#data validation artifact

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str