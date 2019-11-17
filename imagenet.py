from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras import backend as K

def ImageNet(image_path):
    model = ResNet50(weights='imagenet')

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    K.clear_session()

    preds = decode_predictions(preds, top=3)[0]
    pred_str = str(preds[0][1]) + ", " + str(preds[1][1]) + " or " + str(preds[2][1])
    return pred_str
