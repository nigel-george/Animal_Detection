import io
import os

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request,  redirect, url_for
from PIL import Image
import argparse
import torch
import cv2

import time


app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def find_model():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    print("Please place a model file in this directory!")

model_name = find_model()
model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)
model.eval()

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]
    results = model(imgs, size=640)
    return results

def detect_objects(image):
    # Perform object detection
    results = model(image)
    return results

def get_animal_info(animal_name, user_agent):
    url = 'https://en.wikipedia.org/wiki/' + animal_name
    headers = {'User-Agent': user_agent}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        # Concatenate the text from the first few paragraphs
        text = ''.join([p.text for p in paragraphs[:3]])
        return text[:1000]
    else:
        return 'Failed to retrieve information from Wikipedia'



@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return redirect(request.url)

            file = request.files.get('file')
            if not file:
                return render_template('error.html', error_message="No file provided")

            img_bytes = file.read()
            results = get_prediction(img_bytes)

            if results is not None:
                animal_name = results.names[int(results.xyxy[0][0][-1])] if results.xyxy[0].shape[0] > 0 else "Not detected"
                animal_info = get_animal_info(animal_name, 'AnimalSpeciesInformation')
                print(animal_info)
                results.save(save_dir='static')
                filename = 'image0.jpg'
                return render_template('result.html', result_image=filename, model_name=model_name,
                                       animal_name=animal_name, animal_info=animal_info)
            else:
                return render_template('error.html', error_message="Error during prediction")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return render_template('error.html', error_message="An unexpected error occurred")

    return render_template('index.html')

# Add a global variable to control the webcam loop
webcam_active = False

@app.route('/upload_webcam')
def activate_webcam():
    global webcam_active
    cap=cv2.VideoCapture(0)

    # Set the webcam_active flag to True
    webcam_active = True

    while webcam_active:
        ret, frame = cap.read()

        # Convert the OpenCV frame to a PIL image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform object detection on the PIL image
        results = detect_objects(pil_img)

        # Process and visualize the results
        for result in results.pred:
            # Extract bounding box coordinates, class, and confidence
            box = result[:, :4]
            cls = result[:, 5]
            conf = result[:, 4]

            # Draw bounding box with class name and confidence
            for idx in range(len(conf)):
                if conf[idx] > 0.5:  # Display if confidence is greater than 0.5
                    x1, y1, x2, y2 = map(int, box[idx])
                    label = f"{model.names[int(cls[idx])]}: {conf[idx]:.2f}"  # Get class name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                                cv2.LINE_AA)  # Show class and confidence

        # Display the video with bounding boxes
        cv2.imshow('YOLOv7 Object Detection', frame)

        # Check for the key event that closes the OpenCV window (Esc key or X button)
        key = cv2.waitKey(1)

        # Check if the window is closed (X button)
        if key == 27 or cv2.getWindowProperty('YOLOv7 Object Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

        # Add a small delay to avoid high CPU usage
        time.sleep(0.01)

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Use JavaScript to redirect to the index page
    return redirect(url_for('predict'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov7 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)

