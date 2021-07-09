FROM tensorflow/tensorflow:2.5.0-gpu
#TODO change base docker image
# Syntax: # FROM [base_image]

RUN mkdir -p /home

WORKDIR /home

RUN python -m pip install -U pip


COPY requirements.txt /home/
RUN python -m pip install -r requirements.txt

COPY process.py /home/
#TODO copy method components into docker
COPY model_weights_findpvs.h5 /home/
COPY model_architecture_findpvs.json /home/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
#TODO change FindPVS to teamname
LABEL nl.diagnijmegen.rse.algorithm.name=FindPVS

# These labels are required and describe what kind of hardware your algorithm requires to run.
#TODO check that this has the right information (can generate this with evalutils)
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=10G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=8G