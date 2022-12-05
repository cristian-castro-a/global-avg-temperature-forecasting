from pathlib import Path
import pandas as pd


DATA_PATH = Path('../data/pip_dataset.csv')


def main() -> None:
    """
    Exploratory data analysis

    Input:
    - DATA_PATH: Path to raw data
    """
    df = pd.read_csv(DATA_PATH)
    print('test')


if __name__ == '__main__':
    main()