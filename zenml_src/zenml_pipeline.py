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

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from zenml.logger import get_logger
from zenml.pipelines import pipeline

from zenml_src.configs.downloader_config import DownloaderConfig
from zenml_src.configs.trainer_config import TrainerConfig
from zenml_src.materializers.dataset_materializer import DatasetMaterializer
from zenml_src.materializers.model_materializer import ModelMaterializer
from zenml_src.steps.download_raw_data import download_raw_data
from zenml_src.steps.evaluator import evaluator
from zenml_src.steps.generate_dataset import generate_dataset
from zenml_src.steps.trainer import train_cycle_gan

logger = get_logger(__name__)


@pipeline(enable_cache=True)
def cyclegan_pipeline(
        download_data_step,
        generate_dataset_step,
        train_step,
        evaluator_step,
):
    path = download_data_step()
    dataset = generate_dataset_step(dataset_path=path)
    model = train_step(dataset=dataset)
    evaluator_step(model=model)


if __name__ == "__main__":
    step_opt = TrainerConfig(gpu_ids=[0], n_epochs=1, n_epochs_decay=0)
    download_opt = DownloaderConfig(name="maps")

    p = cyclegan_pipeline(
        download_data_step=download_raw_data(opt=download_opt),
        generate_dataset_step=generate_dataset(
            opt=step_opt
        ).with_return_materializers(DatasetMaterializer),
        train_step=train_cycle_gan(
            opt=step_opt
        ).with_return_materializers(ModelMaterializer),
        evaluator_step=evaluator(),
    )
    p.run()
