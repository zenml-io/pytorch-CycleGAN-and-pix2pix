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

from zenml.steps import step

from data import BaseDataset
from models import BaseModel
from util import html
from util.visualizer import save_images
from zenml_src.configs.test_config import TestConfig


@step
def evaluator(
    opt: TestConfig,
    trained_model: BaseModel,
    dataset: BaseDataset,
) -> bool:
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if
    # results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped
    # images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results
    # to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given
    # opt.dataset_mode and other options
    # model = create_model(opt)      # create a model given opt.model and
    # other options
    trained_model.setup(
        opt)  # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase,
                                                                     opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
    opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix.
    # You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses
    # instancenorm without dropout.
    if opt.eval:
        trained_model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our trained_model to opt.num_test images.
            break
        trained_model.set_input(data)  # unpack data from data loader
        trained_model.test()  # run inference
        # visuals = trained_model.get_current_visuals()  # get image results
        visuals = {}
        img_path = trained_model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio,
                    width=opt.display_winsize)
    webpage.save()  # save the HTML

    return True
