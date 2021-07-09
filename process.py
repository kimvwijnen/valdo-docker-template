
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


# TODO change teamname to actual teamname
class Findpvs(SegmentationAlgorithm):
    def __init__(self):
        # TODO change to required input modalities for your method
        self.input_modalities = ['T1', 'T2', 'FLAIR']

        # TODO indicate if uncertainty map should be saved
        self.flag_save_uncertainty = False

        super().__init__(
            # (Skip UniquePathIndicesValidator, because this will error when there are multiple images
            # for the same subject)
            validators=dict(input_image=(UniqueImagesValidator(),)),
            # Indicate with regex which image to load as input, e.g. T1 scan
            file_filters={'input_image':
                          re.compile("/input/sub-.*_space-.*_desc-masked_%s.nii.gz" % self.first_modality)}
        )

        self.first_modality = self.input_modalities[0]

        print("==> Initializing model")

        # --> Load model
        # TODO add code to load model
        weights_path = "/home/model_weights_findpvs.h5"
        architecture_path = "/home/model_architecture_findpvs.json"

        json_file = open(architecture_path, 'r')
        unet_layers = json_file.read()
        json_file.close()

        self.model = tf.keras.models.model_from_json(unet_layers)
        self.model.load_weights(weights_path)

        print("==> Model loaded")

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
        list_results = self.predict(input_images=input_images)

        if self.flag_save_uncertainty:
            assert len(list_results) == 2, "Error, predict function should return a list containing 2 images, " \
                                           "the predicted segmentation and the predicted uncertainty map. " \
                                           "Or change flag_save_uncertainty to False"
        else:
            assert len(list_results) == 1, "Error, predict function should return a list containing 1 image, " \
                                           "only the predicted segmentation. " \
                                           "Or change flag_save_uncertainty to True"

        # Write resulting segmentation to output location
        if not self._output_path.exists():
            self._output_path.mkdir()

        save_description = ['prediction', 'uncertaintymap']
        output_path_list = []

        for i, outimg in enumerate(list_results):
            output_name = Path(input_path_list[0].name.split("desc-masked_")[0] + "%s.nii.gz" % save_description[i])
            segmentation_path = self._output_path / output_name
            print(segmentation_path)
            output_path_list.append(segmentation_path)
            SimpleITK.WriteImage(outimg, str(segmentation_path), True)

        input_name_list = [p.name for p in input_path_list]
        output_name_list = [p.name for p in output_path_list]

        # Write segmentation file path to result.json for this case
        return {
            "outputs": [
                dict(type="metaio_image", filename=output_name_list)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_name_list)
            ],
            "error_messages": [],
        }

    def predict(self, *, input_images: List[SimpleITK.Image]) -> List[SimpleITK.Image]:
        print("==> Running prediction")

        # TODO add code to apply method to input image and output a prediction (and an uncertainty map for task 3 lac)

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
        out_list = []

        # TODO change, see comments
        out_images = [out]
        #  For Task 1 PVS or Task 2 MB: list should contain only the predicted segmentation: [out]
        # out_images = [pred]
        #  For Task 3 Lacunes: list should contain the predicted segmentation and the uncertainty map (in that order!)
        # out_images = [pred, uncertaintymap]

        for outimg in out_images:
            outimg = np.moveaxis(outimg, [0, 2], [2, 0])
            print('Image shape before saving:')
            print(outimg.shape)
            out_itk = SimpleITK.GetImageFromArray(outimg)
            out_itk.CopyInformation(input_images[0])
            assert len(out_itk.GetSize()) == len(input_images[0].GetSize()), "Error, output image should have the" \
                                                                             " same shape as the input image(s)!"
            out_list.append(out_itk)

        print("==> Prediction done")

        # TODO Return a list with a prediction (and an uncertainty map for task 3 lac, the order is important!)
        return out_list


if __name__ == "__main__":
    # TODO change teamname to actual teamname
    Findpvs().process()
