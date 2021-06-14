
import SimpleITK
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

# added imports
import tensorflow as tf


class Findpvs(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
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

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Segment nodule candidates
        segmented_nodules = self.predict(input_image=input_image)

        # Write resulting segmentation to output location
        output_name = Path(input_image_file_path.name.split("desc-masked_")[0] + "desc-prediction.nii.gz")
        segmentation_path = self._output_path / output_name
        if not self._output_path.exists():
            self._output_path.mkdir()
        SimpleITK.WriteImage(segmented_nodules, str(segmentation_path), True)

        # Write segmentation file path to result.json for this case
        return {
            "outputs": [
                dict(type="metaio_image", filename=segmentation_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:

        # image to numpy array
        image = SimpleITK.GetArrayFromImage(input_image)
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

        # --> Apply NN on image
        out = np.squeeze(self.model.predict(image, batch_size=1, verbose=1))
        print('Image shape prediction:')
        print(out.shape)

        # --> Postproc prediction
        # convert numpy array to SimpleITK image again for saving
        out = np.moveaxis(out, [0, 2], [2, 0])
        print('Image shape before saving:')
        print(out.shape)
        out = SimpleITK.GetImageFromArray(out)
        out.CopyInformation(input_image)

        print("==> Prediction done")
        # Prediction is saved in superclass SegmentationAlgorithm, in the process_case function

        return out


if __name__ == "__main__":
    Findpvs().process()
