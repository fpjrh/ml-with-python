"""

Work this to make it execute locally instead of IBM cloud

%%capture
!pip install --no-deps lvis
!pip install tf_slim
!pip install --no-deps tensorflowjs==1.4.0
!pip install tensorflow==1.15.2
!pip install tensorflow_hub
"""
from IPython.display import display, Javascript, Image
import re
import zipfile
from base64 import b64decode
import sys
import os
import json

import io
import random

import tarfile
from datetime import datetime
from zipfile import ZipFile
import six.moves.urllib as urllib
import PIL.Image
from PIL import Image
#
import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util, config_util
from object_detection.utils.label_map_util import get_label_map_dict
from skillsnetwork import cvstudio

import os
if os.path.exists("content-latest.zip"):
    pass
else:
    !wget https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/content/data/content-latest.zip
    !wget https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/content/data/tfrecord.py
    
with zipfile.ZipFile('content-latest.zip', 'r') as zip_ref:
    zip_ref.extractall('')

#################
# Constants
#
CHECKPOINT_PATH = os.path.join(os.getcwd(),'content/checkpoint')
OUTPUT_PATH = os.path.join(os.getcwd(),'content/output')
EXPORTED_PATH = os.path.join(os.getcwd(),'content/exported')
DATA_PATH = os.path.join(os.getcwd(),'content/data')
CONFIG_PATH = os.path.join(os.getcwd(),'content/config')
LABEL_MAP_PATH = os.path.join(DATA_PATH, 'label_map.pbtxt')
TRAIN_RECORD_PATH = os.path.join(DATA_PATH, 'train.record')
VAL_RECORD_PATH = os.path.join(DATA_PATH, 'val.record')

# Initialize the CV Studio Client
cvstudioClient = cvstudio.CVStudio()

# Download All Images
cvstudioClient.downloadAll()

# Get the annotations from CV Studio
annotations = cvstudioClient.get_annotations()
labels = annotations['labels']

os.makedirs(DATA_PATH, exist_ok=True)
with open(LABEL_MAP_PATH, 'w') as f:
    for idx, label in enumerate(labels):
        f.write('item {\n')
        f.write("\tname: '{}'\n".format(label))
        f.write('\tid: {}\n'.format(idx + 1))
        f.write('}\n')
#
from tfrecord import create_tf_record, displaydetectedobject
image_files = [image for image in annotations["annotations"].keys()]

label_map = label_map_util.get_label_map_dict(LABEL_MAP_PATH)

random.seed(42)
random.shuffle(image_files)
num_train = int(0.7 * len(image_files))
train_examples = image_files[:num_train]
val_examples = image_files[num_train:]

create_tf_record(train_examples, annotations["annotations"], label_map, os.path.join(os.getcwd(),'images'), TRAIN_RECORD_PATH)
create_tf_record(val_examples, annotations["annotations"], label_map, os.path.join(os.getcwd(),'images'), VAL_RECORD_PATH)

#
MODEL_TYPE = 'ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18'
CONFIG_TYPE = 'ssd_mobilenet_v1_quantized_300x300_coco14_sync'
download_base = 'http://download.tensorflow.org/models/object_detection/'
model = MODEL_TYPE + '.tar.gz'
tmp = '/resources/checkpoint.tar.gz'

if not (os.path.exists(CHECKPOINT_PATH)):
    # Download the checkpoint
    opener = urllib.request.URLopener()
    opener.retrieve(download_base + model, tmp)

    # Extract all the `model.ckpt` files.
    with tarfile.open(tmp) as tar:
        for member in tar.getmembers():
            member.name = os.path.basename(member.name)
            if 'model.ckpt' in member.name:
                tar.extract(member, path=CHECKPOINT_PATH)
            if 'pipeline.config' in member.name:
                tar.extract(member, path=CONFIG_PATH)

    os.remove(tmp)
# Build the model training pipeline
pipeline_skeleton = 'content/models/research/object_detection/samples/configs/' + CONFIG_TYPE + '.config'
configs = config_util.get_configs_from_pipeline_file(pipeline_skeleton)

label_map = label_map_util.get_label_map_dict(LABEL_MAP_PATH)
num_classes = len(label_map.keys())
meta_arch = configs["model"].WhichOneof("model")

override_dict = {
  'model.{}.num_classes'.format(meta_arch): num_classes,
  'train_config.batch_size': 6,
  'train_input_path': TRAIN_RECORD_PATH,
  'eval_input_path': VAL_RECORD_PATH,
  'train_config.fine_tune_checkpoint': os.path.join(CHECKPOINT_PATH, 'model.ckpt'),
  'label_map_path': LABEL_MAP_PATH
}

configs = config_util.merge_external_params_with_configs(configs, kwargs_dict=override_dict)
pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
config_util.save_pipeline_config(pipeline_config, DATA_PATH)
#
# Start the training
paths = [
    f'home/jupyterlab/conda/envs/python/lib/python3.6',
    f'content/models/research',
    f'content/models/research/slim'
]
os.environ['PYTHONPATH'] = ':'.join(paths)

epochs = 40
start_datetime = datetime.now()
!python -m object_detection.model_main \
    --pipeline_config_path=$DATA_PATH/pipeline.config \
    --num_train_steps=$epochs \
    --num_eval_steps=100

regex = re.compile(r"model\.ckpt-([0-9]+)\.index")
numbers = [int(regex.search(f).group(1)) for f in os.listdir(OUTPUT_PATH) if regex.search(f)]
TRAINED_CHECKPOINT_PREFIX = os.path.join(OUTPUT_PATH, 'model.ckpt-{}'.format(max(numbers)))

!python3 -m object_detection.export_inference_graph \
  --pipeline_config_path=$DATA_PATH/pipeline.config \
  --trained_checkpoint_prefix=$TRAINED_CHECKPOINT_PREFIX \
  --output_directory=$EXPORTED_PATH
end_datetime = datetime.now()

from PIL import Image

# Here you can specify your own image 
URL = 'https://cdn.cliqueinc.com/posts/289533/kamala-harris-face-mask-289533-1602269219518-square.700x0c.jpg' 

with urllib.request.urlopen(URL) as url:
    with open('test.jpg', 'wb') as f:
        f.write(url.read())
image = Image.open('test.jpg')
#
import numpy as np
n,img,accuracy=displaydetectedobject(image)
parameters = {
    'epochs': epochs
}

result = cvstudioClient.report(started=start_datetime, completed=end_datetime, parameters=parameters, accuracy= round(float(accuracy),2)*100)

if result.ok:
    print('Congratulations your results have been reported back to CV Studio!')
#
!tensorflowjs_converter \
  --input_format=tf_frozen_model \
  --output_format=tfjs_graph_model \
  --output_node_names='Postprocessor/ExpandDims_1,Postprocessor/Slice' \
  --quantization_bytes=1 \
  --skip_op_check \
  $EXPORTED_PATH/frozen_inference_graph.pb \
  .
import json

from object_detection.utils.label_map_util import get_label_map_dict

label_map = get_label_map_dict(LABEL_MAP_PATH)
label_array = [k for k in sorted(label_map, key=label_map.get)]

with open(os.path.join('', 'labels.json'), 'w') as f:
    json.dump(label_array, f)

#!cd model_web 
with ZipFile('model_web.zip','w') as zip:
    zip.write('group1-shard1of2.bin')
    zip.write('group1-shard2of2.bin')
    zip.write('model.json')
    zip.write('labels.json')

# Download All Images
cvstudioClient.uploadModel('model_web.zip', {'epochs': epochs })
