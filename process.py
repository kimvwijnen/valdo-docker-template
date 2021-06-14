
import SimpleITK
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import UniqueImagesValidator

# added imports
import tensorflow as tf
from typing import Tuple, List
from pathlib import Path
import re

from evalutils.io import (ImageLoader, SimpleITKLoader)


class Findpvs(SegmentationAlgorithm):
    def __init__(self):
        self.input_modalities = ['T1', 'T2', 'FLAIR']
        self.first_modality = self.input_modalities[0]
        super().__init__(
            # (Skip UniquePathIndicesValidator, because this will error when there are multiple images
            # for the same subject)
            validators=dict(input_image=(UniqueImagesValidator(),)),
            # Indicate with regex which image to load as input, e.g. T1 scan
            file_filters={'input_image':
                          re.compile("/input/sub-.*_space-.*_desc-masked_%s.nii.gz" % self.first_modality)}
        )

        print("==> Initializing model")

        # --> Load model
        weights_path = "/opt/algorithm/model_weights_findpvs.h5"
        architecture_path = "/opt/algorithm/model_architecture_findpvs.json"

        json_file = open(architecture_path, 'r')
        unet_layers = json_file.read()
        json_file.close()

        self.model = tf.keras.models.model_from_json(unet_layers)
        self.model.load_weights(weights_path)

        print("==> Weights loaded")

    def _load_input_image(self, *, case) -> Tuple[List[SimpleITK.Image], List[Path]]:
        input_image_file_path = case["path"]

        input_image_file_loader = self._file_loaders["input_image"]
        if not isinstance(input_image_file_loader, ImageLoader):
            raise RuntimeError(
                "The used FileLoader was not of subclass ImageLoader"
            )
        input_images = []
        input_path_list = []

        # Load the image(s) for this case
        for modality in self.input_modalities:
            # Load all input images, e.g. T1, T2 and FLAIR
            scan_name = Path(input_image_file_path.name.replace('%s.nii.gz' % self.first_modality,
                                                                '%s.nii.gz' % modality))
            modality_path = input_image_file_path.parent / scan_name
            input_images.append(input_image_file_loader.load_image(modality_path))
            input_path_list.append(modality_path)

        # Check that it is the expected image
        if input_image_file_loader.hash_image(input_images[0]) != case["hash"]:
            raise RuntimeError("Image hashes do not match")

        return input_images, input_path_list

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_images, input_path_list = self._load_input_image(case=case)

        # Segment case
        segm_prediction = self.predict(input_images=input_images)

        # Write resulting segmentation to output location
        output_name = Path(input_path_list[0].name.split("desc-masked_")[0] + "desc-prediction.nii.gz")
        segmentation_path = self._output_path / output_name
        if not self._output_path.exists():
            self._output_path.mkdir()
        SimpleITK.WriteImage(segm_prediction, str(segmentation_path), True)

        input_name_list = [p.name for p in input_path_list]
        # Write segmentation file path to result.json for this case
        return {
            "outputs": [
                dict(type="metaio_image", filename=segmentation_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_name_list)
            ],
            "error_messages": [],
        }

    def predict(self, *, input_images: List[SimpleITK.Image]) -> SimpleITK.Image:
        print("==> Running prediction")

        input_list = []

        for img in input_images:
            # image to numpy array
            image = SimpleITK.GetArrayFromImage(img)
            print('Image shape:')
            print(image.shape)

            # --> Preprocess image
            image = np.array(image, dtype=np.float32)

            # normalize image (rescale)
            min_data = np.amin(image)
            max_data = np.amax(image)
            image = (image - min_data) * 1. / (max_data - min_data)

            # convert (z,y,x) to (x,y,z)
            image = np.moveaxis(image, [0, 2], [2, 0])
            print('Image shape after conversion:')
            print(image.shape)

            # add batch axis and channel axis (at front and end for tensorflow)
            image = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)
            input_list.append(image)

        # In TF the last axis contains the channels
        input_channels = np.concatenate(input_list, axis=-1)

        # --> Apply NN on image
        out = np.squeeze(self.model.predict(input_channels, batch_size=1, verbose=1))
        print('Image shape prediction:')
        print(out.shape)

        # --> Postproc prediction
        # convert numpy array to SimpleITK image again for saving
        out = np.moveaxis(out, [0, 2], [2, 0])
        print('Image shape before saving:')
        print(out.shape)
        out = SimpleITK.GetImageFromArray(out)
        out.CopyInformation(input_images[0])

        print("==> Prediction done")
        # Prediction is saved in superclass SegmentationAlgorithm, in the process_case function

        return out


if __name__ == "__main__":
    Findpvs().process()
