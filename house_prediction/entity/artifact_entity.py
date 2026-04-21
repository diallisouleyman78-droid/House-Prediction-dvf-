from dataclasses import dataclass

@dataclass
# Information returned at the end of a pipeline(ingestion,validation,transformation,modeltrainer)

class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str