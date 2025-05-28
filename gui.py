import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN

# Load model
model = tf.keras.models.load_model("emotion2_model.keras")
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize MTCNN
detector = MTCNN()

# GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def Detect(file_path):
    image_bgr = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(image_rgb)

    if not results:
        print("No face detected.")
        label1.configure(foreground="#011638", text="No face detected.")
        show_image(image_rgb)
        return

    # Use first detected face
    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)
    face = image_rgb[y:y + h, x:x + w]
    face_resized = cv2.resize(face, (48, 48))
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
    face_input = face_gray.astype("float32") / 255.0
    face_input = np.expand_dims(face_input, axis=0)
    face_input = np.expand_dims(face_input, axis=-1)

    prediction = model.predict(face_input)
    top_index = np.argmax(prediction)
    top_emotion = EMOTIONS_LIST[top_index]

    # Print top 3 probabilities
    print("\nTop 3 Emotions with Probabilities:")
    sorted_indices = np.argsort(prediction[0])[::-1][:3]
    for idx in sorted_indices:
        print(f"{EMOTIONS_LIST[idx]}: {prediction[0][idx]:.2f}")

    label1.configure(foreground="#011638", text=f"Emotion: {top_emotion}")

    # Draw rectangle and show
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    show_image(image_rgb)

def show_image(img_rgb):
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img_pil)
    sign_image.configure(image=img_tk)
    sign_image.image = img_tk

def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    uploaded = Image.open(file_path)
    uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
    im = ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image = im
    label1.configure(text='')
    show_Detect_button(file_path)

# GUI layout
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')

heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()

