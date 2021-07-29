# valdo-docker-example

PyTorch Example for the [Where is VALDO challenge](https://valdo.grand-challenge.org/)

For a description of how to prepare your submission see the [Prepare Docker page](https://valdo.grand-challenge.org/Docker/)

This is an example submission. See the main branch for the template to fill in to submit your own method.
In this example a U-Net is used with random weights (so no trained weights are loaded). When adapting the code for your own method do not forget to copy in your weights in the Dockerfile and load the weights in the init function in process.py.

This example was produced using inspiration from the [vesselSegmentor](https://github.com/DIAGNijmegen/drive-vessels-unet/tree/master/vesselSegmentor) repo. 

# ValdoTorch Algorithm

The source code for the algorithm container for
ValdoTorch, generated with
evalutils version 0.2.4.

