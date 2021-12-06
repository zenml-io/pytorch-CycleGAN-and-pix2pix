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

import torch
from zenml.logger import get_logger
from zenml.pipelines import pipeline

from zenml_src.configs.trainer_config import TrainerConfig
from zenml_src.configs.downloader_config import DownloaderConfig
from zenml_src.materializers.cycle_gan_materializer import CycleGanMaterializer
from zenml_src.steps.downloader import downloader

from zenml_src.steps.generate_dataset import generate_dataset
from zenml_src.steps.generate_model import generate_model
from zenml_src.steps.trainer import train_cycle_gan
from zenml.steps.step_context import StepContext

logger = get_logger(__name__)


def prestep(context: StepContext, opt: TrainerConfig):
    # create a dataset given opt.dataset_mode and other options
    opt.checkpoints_dir = context.get_output_artifact_uri()

    # zenml additions
    opt.dataroot = path  # need to do this to get the rest of the code to work
    if opt.gpu_ids:
        torch.cuda.set_device(opt.gpu_ids[0])
    opt.print_options()
    return opt


@pipeline(enable_cache=True)
def cyclegan_pipeline(
        download_data_step,
        generate_dataset_step,
        generate_model_step,
        train_step
):
    path = download_data_step()
    dataset = generate_dataset_step()
    model = generate_model_step()
    train_step(path=path, model=model, dataset=dataset)


if __name__ == "__main__":
    step_opt = TrainerConfig(gpu_ids=[], n_epochs=1, n_epochs_decay=0)

    p = cyclegan_pipeline(
        download_data_step=downloader(DownloaderConfig(name="maps")),
        generate_model_step=generate_model(
            opt=step_opt
        ).with_return_materializers(CycleGanMaterializer),
        generate_dataset_step=generate_dataset(
            opt=step_opt
        ).with_return_materializers(CycleGanMaterializer),
        train_step=train_cycle_gan(
            opt=step_opt
        ).with_return_materializers(CycleGanMaterializer),
    )
    p.run()
