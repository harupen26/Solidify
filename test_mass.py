import requests
from PIL import Image
import io

# Create a dummy image
img = Image.new('RGB', (100, 100), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

url = 'http://127.0.0.1:8000/optimize'
files = {'file': ('test.png', img_byte_arr, 'image/png')}
data = {
    'handle_x': '0.5', 
    'handle_y': '0.5',
    'target_mass': '1.5',
    'ball_mass': '0.3'
}

try:
    print("Sending request with Mass Inputs...")
    response = requests.post(url, files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
