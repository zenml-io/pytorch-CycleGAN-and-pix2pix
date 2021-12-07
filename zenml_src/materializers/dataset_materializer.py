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
from typing import Any, Type

from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from data import BaseDataset, create_dataset
from zenml_src.configs.trainer_config import TrainerConfig

DATASET_OPT_DUMP = "dataset_opt_dump.json"


class DatasetMaterializer(BaseMaterializer):
    """Materializer to read/write Pytorch models."""

    ASSOCIATED_TYPES = [BaseDataset]

    def handle_input(self, data_type: Type[Any]) -> BaseDataset:
        """Reads and returns a BaseDataset.

        Returns:
            A loaded BaseDataset.
        """
        super().handle_input(data_type)
        opt = TrainerConfig.parse_file(
            os.path.join(self.artifact.uri, DATASET_OPT_DUMP)
        )
        dataset = create_dataset(opt)
        return dataset

    def handle_return(self, dataset: BaseDataset) -> None:
        """Writes a BaseDataset to disk.

        Args:
            dataset: A BaseDataset instance.
        """
        super().handle_return(dataset)
        fileio.write_file_contents_as_string(
            os.path.join(self.artifact.uri, DATASET_OPT_DUMP),
            dataset.opt.json()
        )
