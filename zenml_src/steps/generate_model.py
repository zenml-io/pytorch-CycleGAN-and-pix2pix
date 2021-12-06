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

from models import BaseModel, create_model
from zenml_src.configs.trainer_config import TrainerConfig
from zenml_src.zenml_pipeline import prestep


@step
def generate_model(opt: TrainerConfig) -> BaseModel:
    """Generates a model"""
    # create a model given opt.model and other options
    opt = prestep(opt)
    model = create_model(opt)
    return model