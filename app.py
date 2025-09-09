import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load model
model = tf.keras.models.load_model("mnist_model.h5")

st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload an image of a digit (0-9) to predict.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    image = ImageOps.invert(image)  # invert colors if needed
    image = image.resize((28,28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1,28,28,1)

    # Show image
    st.image(image, caption="Uploaded Digit", use_container_width=True)

    # Prediction
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.write(f"### ✅ Predicted Digit: {digit}")
