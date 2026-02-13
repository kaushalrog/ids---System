import requests
import time

print("Testing Flask connection...")
try:
    for i in range(5):
        try:
            resp = requests.get('http://127.0.0.1:5000/health', timeout=2)
            print(f"Attempt {i+1}: Status {resp.status_code}")
            break
        except requests.exceptions.ConnectionError as e:
            print(f"Attempt {i+1}: Connection refused - {e}")
            time.sleep(1)
except Exception as e:
    print(f"Error: {e}")
