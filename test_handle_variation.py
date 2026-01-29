import requests
from PIL import Image
import io
import json

# Create a dummy image
img = Image.new('RGB', (100, 100), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)
img_bytes = img_byte_arr.getvalue()

url = 'http://127.0.0.1:8000/optimize'

# Case 1: Handle at 0.0, 0.0
files1 = {'file': ('test.png', img_bytes, 'image/png')}
data1 = {
    'handle_x': '0.0', 
    'handle_y': '0.0',
    'target_mass': '0.8',
    'ball_mass': '0.2'
}
print("--- Request 1 (Handle 0.0, 0.0) ---")
res1 = requests.post(url, files=files1, data=data1).json()
d1 = res1['optimized']['dist']
print(f"Dist 1: {d1}")

# Case 2: Handle at 0.5, 0.5
files2 = {'file': ('test.png', img_bytes, 'image/png')}
data2 = {
    'handle_x': '0.5', 
    'handle_y': '0.5',
    'target_mass': '0.8',
    'ball_mass': '0.2'
}
print("--- Request 2 (Handle 0.5, 0.5) ---")
res2 = requests.post(url, files=files2, data=data2).json()
d2 = res2['optimized']['dist']
print(f"Dist 2: {d2}")

if abs(d1 - d2) < 0.0001:
    print("FAIL: Distance did not change!")
else:
    print("SUCCESS: Distance changed.")
