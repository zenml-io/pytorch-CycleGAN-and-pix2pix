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

import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import requests
import torch
from zenml.logger import get_logger
from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.steps.base_step_config import BaseStepConfig

from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

logger = get_logger(__name__)


def download_file(url: str, out_file_name: str):
    with open(out_file_name, "wb") as f:
        logger.info("Downloading %s" % out_file_name)
        response = requests.get(url, stream=True)
        total_length = response.headers.get("content-length")

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                sys.stdout.flush()


def download_extract_zipfile(url, save_path):
    """Download and extract a zipfile"""
    # Download the file from the URL
    # Create a new file on the hard drive
    tempzip = NamedTemporaryFile()

    download_file(url, tempzip.name)

    # Re-open the newly-created file with ZipFile()
    zf = ZipFile(tempzip.name)
    # Extract its contents into <extraction_path>
    # note that extractall will automatically create the path
    zf.extractall(path=save_path)
    # close the ZipFile instance
    zf.close()
    # Close the zup file
    tempzip.close()


@step
def download_maps() -> str:
    """Download the MOT20 dataset."""
    url = "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/maps.zip"
    download_path = Path("./datasets")
    download_path.mkdir(parents=True, exist_ok=True)
    download_extract_zipfile(url=url, save_path=download_path)
    return str(download_path.absolute() / 'maps')


class BaseConfig(BaseStepConfig):
    dataroot: str = ""
    name: str = "experiment_name"
    use_wandb: bool = True
    gpu_ids: str = '-1'
    checkpoints_dir: str = './checkpoints'
    model: str = 'cycle_gan'
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    netD: str = 'basic'
    netG: str = 'resnet_9blocks'
    n_layers_D: int = 3
    norm: str = "instance"
    init_type: str = "normal"
    init_gain: float = 0.02
    no_dropout: bool = True
    dataset_mode: str = "unaligned"
    direction: str = "AtoB"
    serial_batches: bool = True
    num_threads: int = 4
    batch_size: int = 1
    load_size: int = 286
    crop_size: int = 256
    max_dataset_size: float = float("inf")
    preprocess: str = "resize_and_crop"
    no_flip: bool = True
    display_winsize: int = 256
    epoch: str = "latest"
    load_iter: str = "0"
    verbose: bool = True
    suffix: str = ""


class TrainerConfig(BaseConfig):
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
    save_epoch_freq: int = 5
    save_by_iter: bool = True
    continue_train: bool = True
    epoch_count: int = 1
    phase: str = "train"
    n_epochs: int = 100
    n_epochs_decay: int = 100
    beta1: float = 0.5
    lr: float = 0.0002
    gan_mode: str = "lsgan"
    pool_size: int = 50
    lr_policy: str = "linear"
    lr_decay_iters: int = 50


@step
def train_cycle_gan(
        opt: TrainerConfig,
        path: str,
) -> torch.nn.Module:
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # create a model given opt.model and other options
    model = create_model(opt)

    # regular setup: load and print networks; create schedulers
    model.setup(opt)

    # create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)

    total_iters = 0  # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # for different epochs; we save the model by <epoch_count>,
        # <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch,
        # reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the
        # results to HTML at least once every epoch
        model.update_learning_rate()  # update learning rates in the
        # beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(
                data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions,
            # get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on
                # visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(),
                                                   epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses
                # and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses,
                                                t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(
                        epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest
                # model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (
                    epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter \
                    else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:  # cache our model every
            # <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (
                epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay,
            time.time() - epoch_start_time))
    return model


@pipeline(enable_cache=True)
def cyclegan_pipeline(download_data_step, train_step):
    path = download_data_step()
    train_step(path=path)


p = cyclegan_pipeline(
    download_data_step=download_maps(),
    train_step=train_cycle_gan(),
)
p.run()
