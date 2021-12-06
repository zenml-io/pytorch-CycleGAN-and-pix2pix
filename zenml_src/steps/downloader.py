#  Copyright (c) ZenML GmbH 2021. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from pathlib import Path

from zenml.steps import step

from zenml_src.utils import download_extract_zipfile
from zenml_src.configs.downloader_config import DownloaderConfig


@step
def downloader(config: DownloaderConfig) -> str:
    """Download the MOT20 dataset."""
    url = f"http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/" f"" \
          f"{config.name}.zip"
    download_path = Path(config.path)
    download_path.mkdir(parents=True, exist_ok=True)
    download_extract_zipfile(url=url, save_path=download_path)
    return str(download_path.absolute() / config.name)