import os
import zipfile
from pathlib import Path

import kaggle
import pandas as pd


def download_data(
    competition_name: str, path: Path, force_download: bool = False, unzip: bool = False
):
    kaggle.api.competition_download_cli(
        competition_name,
        path=path,
        force=force_download,
    )
    if unzip:
        with zipfile.ZipFile(path / f"{competition_name}.zip", "r") as zip_ref:
            zip_ref.extractall(path)

    os.system(f"du -h {path}")


def create_submission(
    df: pd.DataFrame,
    competition_name: str,
    id_col: str = "image_id",
    label_col: str = "label",
    output_file: str = "submission.csv",
    should_submit: bool = False,
    submission_message: str = "Fingers crossed",
):
    assert hasattr(df, id_col), f"Dataframe missing column: {id_col}"
    assert hasattr(df, label_col), f"Dataframe missing column: {label_col}"

    submission = submission.sort_values(id_col)
    submission.to_csv(output_file, index=False)

    if should_submit:
        kaggle.api.competition_submit(output_file, submission_message, competition_name)

    return submission
