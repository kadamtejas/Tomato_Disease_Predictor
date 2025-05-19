from fastapi import FastAPI,File,UploadFile
import tensorflow as tf
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

MODEL = tf.keras.models.load_model(r"D:\desktop\NeoSoft\SL\Tomato_Disease_Predictor\model.h5")
CLASS_NAMES = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

@app.get("/ping")
async def ping():
    return "Hello, I am learning"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file:UploadFile=File(...)
):
    image =read_file_as_image(await file.read())

    image_batch = np.expand_dims(image,0)

    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.argmax(predictions[0])

    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }
    


if __name__ == "__main__":
    uvicorn.run(app,host='localhost', port = 8000)