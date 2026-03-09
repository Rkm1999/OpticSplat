import requests
import os

url = "http://127.0.0.1:8000"
test_image = "backend/uploads/0a6f0992-e866-460d-aac7-0b2fa043770b/input.jpg"

def test_history():
    print("Testing /history...")
    try:
        r = requests.get(f"{url}/history")
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print(f"History items: {len(r.json())}")
            return True
    except Exception as e:
        print(f"Error: {e}")
    return False

def test_generate():
    print(f"Testing /generate with {test_image}...")
    if not os.path.exists(test_image):
        print("Test image not found.")
        return False
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            r = requests.post(f"{url}/generate", files=files)
            print(f"Status: {r.status_code}")
            if r.status_code == 200:
                print(f"Generated: {r.json()['id']}")
                return True
            else:
                print(f"Error: {r.text}")
    except Exception as e:
        print(f"Error: {e}")
    return False

if __name__ == "__main__":
    # Check if server is running
    try:
        requests.get(url)
    except:
        print("Server not running. Please start it with run_simulator.bat")
        exit(1)
        
    test_history()
    # test_generate() # Skipping generate by default as it's slow/heavy
