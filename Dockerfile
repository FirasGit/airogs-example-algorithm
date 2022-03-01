# Edit the base image here, e.g., to use 
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/) 
# or a PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Needed for opencv
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y git

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

# Install required packages
# e.g. `python -m pip install sklearn==...`

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user --upgrade pip
RUN python -m pip install --user -rrequirements.txt
RUN python -m pip show imagecodecs

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm do_inference.py /opt/algorithm/

# TODO: For new version, change this
# Copy network folder
RUN mkdir /opt/algorithm/networks/
COPY --chown=algorithm:algorithm ./networks/ /opt/algorithm/networks/

RUN python -m pip install --user -r ./networks/requirements.txt

# Needed for Flow
RUN python -m pip install --user git+https://github.com/VLL-HD/FrEIA.git

RUN mkdir -p /home/algorithm/.config/Ultralytics/
COPY --chown=algorithm:algorithm ./networks/detection/Arial.ttf /home/algorithm/.config/Ultralytics/

# Needed for Olis Method
RUN mkdir -p /home/algorithm/.cache/torch/hub/checkpoints
COPY --chown=algorithm:algorithm ./networks/anomaly_detection/efficientnet-b4-6ed6700e.pth /home/algorithm/.cache/torch/hub/checkpoints/

# Needed for Flow Method
COPY --chown=algorithm:algorithm ./networks/anomaly_detection_flow/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth /home/algorithm/.cache/torch/hub/checkpoints/

# Copy additional files, such as model weights
# e.g. `COPY --chown=algorithm:algorithm weights.pth /opt/algorithm/weights.pth`

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=airogs_algorithm

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=1G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=0
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=


