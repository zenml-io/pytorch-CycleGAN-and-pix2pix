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

from zenml.io.utils import write_file_contents_as_string
from zenml.materializers.base_materializer import BaseMaterializer

from models import create_model
from models.base_model import BaseModel
from zenml_src.configs.trainer_config import TrainerConfig

MODEL_OPT_DUMP = "model_opt_dump.json"


class ModelMaterializer(BaseMaterializer):
    """Materializer to read/write Pytorch models."""

    ASSOCIATED_TYPES = [BaseModel]

    def handle_input(self, data_type: Type[Any]) -> BaseModel:
        """Reads and returns a BaseModel.

        Returns:
            A loaded BaseModel.
        """
        super().handle_input(data_type)
        opt = TrainerConfig.parse_file(
            os.path.join(self.artifact.uri, MODEL_OPT_DUMP)
        )
        model = create_model(opt)
        model.load_networks("latest")
        return model

    def handle_return(self, model: BaseModel) -> None:
        """Writes a BaseModel to disk.

        Args:
            model: A BaseModel instance.
        """
        super().handle_return(model)
        write_file_contents_as_string(
            os.path.join(self.artifact.uri, MODEL_OPT_DUMP),
            model.opt.json()
        )
        model.save_networks("latest")
