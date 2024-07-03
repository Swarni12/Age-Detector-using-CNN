import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('age_prediction_model.h5')

# Function to preprocess the selected image
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", "Failed to load the image.")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (48, 48))
        image = image / 255.0
        return image
    except Exception as e:
        messagebox.showerror("Error", f"Error processing image: {e}")
        return None

# Function to predict age from the selected image
def predict_age(image_path):
    image = preprocess_image(image_path)
    if image is not None:
        image_test = np.expand_dims(image, axis=0)  # Add batch dimension
        pred_l = model.predict(image_test)
        age = int(np.round(pred_l[0][0]))
        return age
    return None

# Function to open a file dialog and select an image file
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
    if file_path:
        age = predict_age(file_path)
        if age is not None:
            image = Image.open(file_path)
            image = image.resize((250, 250))
            photo = ImageTk.PhotoImage(image)
            
            label_image.config(image=photo)
            label_image.photo = photo
            
            label_result.config(text=f"Predicted Age: {age}")
        else:
            messagebox.showerror("Error", "Failed to predict age.")
    else:
        messagebox.showwarning("Warning", "No image selected.")

# GUI setup
root = tk.Tk()
root.title("Age Detector")

# Button to select an image
btn_select = tk.Button(root, text="Select Image", command=select_image)
btn_select.pack(pady=10)

# Label to display the selected image
label_image = tk.Label(root)
label_image.pack()

# Label to display the predicted age
label_result = tk.Label(root, text="Predicted Age: -", font=("Helvetica", 16))
label_result.pack(pady=10)

# Run the main event loop
root.mainloop()