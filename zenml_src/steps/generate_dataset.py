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

from zenml.steps import step

from data import BaseDataset, create_dataset
from zenml_src.configs.trainer_config import TrainerConfig


@step
def generate_dataset(opt: TrainerConfig, dataset_path: str) -> BaseDataset:
    """Generates a dataset"""
    # need to do this to get the rest of the code to work
    opt.dataroot = dataset_path
    dataset = create_dataset(opt)
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print("The number of training images = %d" % dataset_size)
    return dataset
