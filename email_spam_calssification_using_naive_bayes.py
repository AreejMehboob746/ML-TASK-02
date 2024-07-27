# -*- coding: utf-8 -*-
"""Email Spam Calssification Using Naive Bayes

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/email-spam-calssification-using-naive-bayes-24ae1fac-e2f4-49dc-adf0-e215ae47f26c.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20240715/auto/storage/goog4_request%26X-Goog-Date%3D20240715T164928Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D2d1c5d5293b12748f889711f46f57eaa77d0dc511533435a45be14e7489090ada5dede9571a3ac8b2ac0b7555cdb15b9c5fe7cc8ae36c41a2d79ddf8c1bb42f551ebd07f6cba94da497cf5856240a985355aae029e355c2c7dee3b800a8ece1e3214bc770b698bf3e8c01951b86219cafe038d282641059b297dda8e899e07698cce3d03cc2bb8d84b9966eacd370fefa1c33824249fa6fa6bfe80a4c77fc6450381ea8a4c62b3fc760ae70fcd903737198c1ee88666466dc1e6b609bd307039ff8babbdc48a3cf2390e8702fa21f0343a72227b2c43fe1d4d1a82eba5d14ae086dacae3c386b8dac1cb7fd429a547f826b188547fc9957d1f8191eced7ace7a
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'email-spam-classification-dataset-csv:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F547699%2F998616%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240715%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240715T164928Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D51615928baa40c1ac47f2e93a8bd01f065dd96479356dd4e775852fbbdedce061b186fcb96296b7f3f61110b14afa009cbf9947b760aa241e3f611fb1762f0c6eb1beafe07c34108f0e9b0f20516f591a573edbf20360d91dd8a8e342b6c117c50eda7225a025160fafbfb85a3922aee0e46cfadba455ebca62b6a9647b4e3d078e6967fdc30caad189c7fc95a8966118e4c3eb58ed8174da4a6245459f30e1b6b58b568b4b4c18912abe85db0c423b5d0ce41b3e102c8a575f5a15939e526c54ad260aa9c34e331c485bf02a1c542c5c265ec6dd062306ac0dacced6925723754553efb24ab95f85ed010e997bd7055c94612abe4c18c188a90bdc2df33d0cb'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/email-spam-classification-dataset-csv/emails.csv')
data.head()

data.info()

data.drop(columns=['Email No.'] , inplace = True)

data.head()

data.isna().sum()

data.shape

x = data.iloc[: , 0:3000]
y = data.iloc[: , 3000]

(x.shape , y.shape)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=40)

# Build Model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
(np.array(y_test),y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy : " , np.round(accuracy_score(y_test , y_pred),4)*100)

# Cross Validation
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(model,x_train , y_train , cv=10)
cross_validation.mean()*100