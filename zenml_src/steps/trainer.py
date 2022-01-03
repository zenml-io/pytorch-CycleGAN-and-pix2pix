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

import time

import torch
from zenml.steps import step
from zenml.steps.step_context import StepContext

from data import BaseDataset
from models import BaseModel
from models import create_model
from zenml_src.configs.trainer_config import TrainerConfig


@step(enable_cache=True)
def train_cycle_gan(
        context: StepContext,
        dataset: BaseDataset,
        opt: TrainerConfig,
) -> BaseModel:
    opt.checkpoints_dir = context.get_output_artifact_uri()
    if opt.gpu_ids:
        torch.cuda.set_device(opt.gpu_ids[0])
    opt.print_options()
    # regular setup: load and print networks; create schedulers
    model = create_model(opt)
    model.setup(opt)

    total_iters = 0  # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # for different epochs; we save the model by <epoch_count>,
        # <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch,
        # reset to 0 every epoch
        # visualizer.reset()  # reset the visualizer: make sure it saves the
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
                # visualizer.display_current_results(
                # model.get_current_visuals(),
                #                                    epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses
                # and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                if opt.display_id > 0:
                    pass
                    # visualizer.plot_current_losses(epoch, float(
                    #     epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest
                # model every <save_latest_freq> iterations
                print(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                save_suffix = "iter_%d" % total_iters if opt.save_by_iter \
                    else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:  # cache our model every
            # <save_epoch_freq> epochs
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.n_epochs + opt.n_epochs_decay,
               time.time() - epoch_start_time)
        )
    return model
