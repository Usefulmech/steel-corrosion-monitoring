"""
Test RTSP connection before running full monitoring
"""

import cv2

def test_rtsp(rtsp_url):
    """Test if RTSP stream is accessible"""
    print(f"Testing connection to: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("❌ Connection failed!")
        print("\nCheck:")
        print("  1. Camera is powered on and on network")
        print("  2. RTSP URL is correct")
        print("  3. Username/password are correct")
        print("  4. Port 554 is not blocked by firewall")
        return False
    
    # Try to read one frame
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Connected but cannot read frames!")
        cap.release()
        return False
    
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print("✅ Connection successful!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frame shape: {frame.shape}")
    
    # Save test frame
    cv2.imwrite("rtsp_test_frame.jpg", frame)
    print("   Test frame saved as: rtsp_test_frame.jpg")
    
    cap.release()
    return True

if __name__ == "__main__":
    # Replace with your camera's RTSP URL
    RTSP_URL = "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"
    
    test_rtsp(RTSP_URL)