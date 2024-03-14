
# def fetch_cat_image():
#     url = "https://api.thecatapi.com/v1/images/search"
#     response = requests.get(url)
#     data = json.loads(response.text)
#     image_url = data[0]['url']
#     return image_url

# cat_image_url = fetch_cat_image()
# print("Here's a random cat image:")
# print(cat_image_url)
!pip install numpy

!pip install opencv-python

import requests
import os
import cv2
import numpy as np

api_key = os.environ.get('CATS_API_KEY')
response = requests.get("https://api.thecatapi.com/v1/images/search", headers={"x-api-key": api_key})
cat_image_url = response.json()[0]['url']
print(cat_image_url)

def check_for_cat(image_url):
    # Download the image
    response = requests.get(image_url)
    image_data = response.content

    # Convert image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Load the pre-trained cat cascade classifier
    cat_cascade = cv2.CascadeClassifier('path/to/cat_cascade.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect cats in the image
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any cats are detected
    if len(cats) > 0:
        return True
    else:
        return False

# Usage example
has_cat = check_for_cat(cat_image_url)
if has_cat:
    print("The image contains a cat!")
else:
    print("The image does not contain a cat.")
