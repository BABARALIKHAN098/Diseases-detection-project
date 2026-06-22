from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    test_size: float = 0.2

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def run(self):
        # implement train/test split and ingestion
        return None
