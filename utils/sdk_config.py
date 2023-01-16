from pathlib import Path


class SDKConfig:
    """
    SDK for this project
    """
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.root_dir / 'data'
        self.tmp_dir = self.root_dir / 'tmp'

    def get_output_dir(self, output_dir: str) -> Path:
        output_dir = self.tmp_dir / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir