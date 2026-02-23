# test_minimal.py
from fastapi import FastAPI
import uvicorn
import socket

app = FastAPI()

@app.get("/")
def root():
    return {"message": "MINIMAL API WORKING"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTING MINIMAL API")
    print("="*50)
    
    # Check if port is available
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 8000))
    if result == 0:
        print("❌ Port 8000 is already in use!")
        sock.close()
    else:
        print("✅ Port 8000 is available")
    sock.close()
    
    print("\nStarting server...")
    print("Try: http://127.0.0.1:8000")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)