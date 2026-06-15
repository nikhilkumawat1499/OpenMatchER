from pathlib import Path

import pandas as pd
from fastapi import UploadFile

from app.core.config import get_settings


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError("Only CSV, Parquet, and JSON datasets are supported")


async def persist_upload(project_id: str, upload: UploadFile) -> tuple[Path, pd.DataFrame]:
    settings = get_settings()
    directory = Path(settings.upload_dir) / project_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / upload.filename
    path.write_bytes(await upload.read())
    return path, read_dataframe(path)
