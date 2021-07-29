FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
#TODO change base docker image
# Syntax: # FROM [base_image]

#TODO fill in all ...

RUN mkdir -p /home

WORKDIR /home

RUN python -m pip install -U pip


COPY requirements.txt /home/
RUN python -m pip install -r requirements.txt

COPY process.py /home/
#TODO copy method components into docker
# (In this example a U-Net with random weights was used,
# for your own method don't forget to copy your weights into the Docker container here)
# COPY best_metric_model_segmentation2d_dict.pth /home/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
#TODO change ValdoTorch to teamname
LABEL nl.diagnijmegen.rse.algorithm.name=ValdoTorch

# These labels are required and describe what kind of hardware your algorithm requires to run.
#TODO check that this has the right information (can generate this with evalutils)
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=10G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=8G