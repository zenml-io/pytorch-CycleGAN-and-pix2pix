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
from typing import Any, Type, Union

import torch
from torch.nn import Module  # type: ignore[attr-defined]
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.types.pytorch_types import TorchDict

DEFAULT_FILENAME = "entire_model.pt"


class CycleGanMaterializer(BaseMaterializer):
    """Materializer to read/write Pytorch models."""

    ASSOCIATED_TYPES = [Module, TorchDict]

    def handle_input(self, data_type: Type[Any]) -> Union[Module, TorchDict]:
        """Reads and returns a PyTorch model.

        Returns:
            A loaded pytorch model.
        """
        super().handle_input(data_type)
        raise NotImplementedError

    def handle_return(self, model: Union[Module, TorchDict]) -> None:
        """Writes a PyTorch model.

        Args:
            model: A torch.nn.Module or a dict to pass into model.save
        """
        super().handle_return(model)
        for name in model.model_names:
            if isinstance(name, str):
                save_filename = 'final_net_%s.pth' % name
                save_path = os.path.join(self.artifact.uri, self.save_dir,
                                         save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
