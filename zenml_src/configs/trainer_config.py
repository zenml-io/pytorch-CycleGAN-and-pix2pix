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
from typing import List

from zenml.steps.base_step_config import BaseStepConfig

from util import util


class BaseConfig(BaseStepConfig):
    name: str = "maps_cyclegan"
    use_wandb: bool = False
    gpu_ids: List[int] = []
    checkpoints_dir: str = "../checkpoints"
    model: str = "cycle_gan"
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    netD: str = "basic"
    netG: str = "resnet_9blocks"
    n_layers_D: int = 3
    norm: str = "instance"
    init_type: str = "normal"
    init_gain: float = 0.02
    no_dropout: bool = True
    dataset_mode: str = "unaligned"
    direction: str = "AtoB"
    serial_batches: bool = True
    num_threads: int = 0
    batch_size: int = 1
    load_size: int = 286
    crop_size: int = 256
    max_dataset_size: float = float("inf")
    preprocess: str = "resize_and_crop"
    no_flip: bool = True
    display_winsize: int = 256
    epoch: str = "latest"
    load_iter: int = 0
    verbose: bool = True
    suffix: str = ""


class BaseTrainerConfig(BaseConfig):
    dataroot: str = ""
    isTrain: bool = True
    display_freq: int = 400
    display_ncols: int = 4
    display_id: int = 1
    display_server: str = "http://localhost"
    display_env: str = "main"

    display_port: int = 8097
    update_html_freq: int = 1000
    print_freq: int = 100
    no_html: bool = True
    save_latest_freq: int = 5000
    save_epoch_freq: int = 0
    save_by_iter: bool = True
    continue_train: bool = False
    epoch_count: int = 0
    phase: str = "train"
    n_epochs: int = 0
    n_epochs_decay: int = 0
    beta1: float = 0.5
    lr: float = 0.0002
    gan_mode: str = "lsgan"
    pool_size: int = 50
    lr_policy: str = "linear"
    lr_decay_iters: int = 50


class TrainerConfig(BaseTrainerConfig):
    lambda_A: float = 10.0
    lambda_B: float = 10.0
    lambda_identity: float = 0.5

    def print_options(self):
        """Print and save options

        It will print current options .
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = "----------------- Options ---------------\n"
        for k, v in sorted(self.dict().items()):
            message += "{}\n".format(str(k), str(v))
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(self.checkpoints_dir, self.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "{}_opt.txt".format(self.phase))
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")
