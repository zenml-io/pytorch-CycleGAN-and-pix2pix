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
import sys
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import requests

from zenml_src.zenml_pipeline import logger


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
    tempzip = NamedTemporaryFile(delete=False)

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
    os.unlink(tempzip.name)