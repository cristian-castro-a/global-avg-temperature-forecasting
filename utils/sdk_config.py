from pathlib import Path


class SDKConfig:
    """
    SDK for this project
    """
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.root_dir / 'data'
        self.eda_dir = self.root_dir / 'exploratory_data_analysis'
        self.models_dir = self.root_dir / 'models'
        self.base_model_dir = self.models_dir / 'base_model'
        self.univar_lstm_dir = self.models_dir / 'univar_lstm_model'
        self.multivar_lstm_dir = self.models_dir / 'multivar_lstm_model'
        self.tmp_dir = self.root_dir / 'tmp'

    def get_output_dir(self, output_dir: str) -> Path:
        output_dir = self.tmp_dir / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def set_working_dirs(self):
        for key, value in self.__dict__.items():
            if key is not 'root_dir':
                value.mkdir(parents=True, exist_ok=True)