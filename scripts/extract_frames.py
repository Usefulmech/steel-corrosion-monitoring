"""
Extract frames from CCTV video for annotation
DO NOT annotate videos frame-by-frame - extract first, then annotate images!
"""

import cv2
import os

def extract_frames_from_video(video_path, output_dir, frame_interval=60):
    """
    Extract frames from video at intervals
    
    Args:
        video_path: Path to video file (.mp4, .avi, etc.)
        output_dir: Where to save extracted frames
        frame_interval: Extract every Nth frame 
                       (60 = 1 frame every 2 seconds at 30fps)
    
    Example:
        extract_frames_from_video(
            "cctv_corrosion.mp4", 
            "extracted_frames",
            frame_interval=90  # 1 frame every 3 seconds
        )
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video Info:")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Extracting every {frame_interval} frames...")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every Nth frame
        if frame_count % frame_interval == 0:
            output_path = os.path.join(
                output_dir, 
                f"frame_{saved_count:04d}.jpg"
            )
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            if saved_count % 50 == 0:
                print(f"   Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n✅ Extraction Complete!")
    print(f"   Total frames extracted: {saved_count}")
    print(f"   Saved to: {output_dir}")
    print(f"   Now upload these images to Roboflow for annotation!")


def extract_frames_from_rtsp(rtsp_url, output_dir, duration_seconds=300, 
                             frame_interval=60):
    """
    Extract frames directly from RTSP stream
    
    Args:
        rtsp_url: RTSP URL of camera
        output_dir: Where to save frames
        duration_seconds: How long to record (default: 5 minutes)
        frame_interval: Extract every Nth frame
    
    Example:
        extract_frames_from_rtsp(
            "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101",
            "rtsp_frames",
            duration_seconds=600,  # 10 minutes
            frame_interval=90
        )
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📡 Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("❌ Could not connect to RTSP stream!")
        return
    
    print(f"✅ Connected! Recording for {duration_seconds} seconds...")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # Default to 25 if not available
    max_frames = int(duration_seconds * fps)
    
    frame_count = 0
    saved_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Lost connection, stopping...")
            break
        
        if frame_count % frame_interval == 0:
            output_path = os.path.join(
                output_dir,
                f"rtsp_frame_{saved_count:04d}.jpg"
            )
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            if saved_count % 20 == 0:
                print(f"   Captured {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"\n✅ Capture complete! Saved {saved_count} frames to {output_dir}")


if __name__ == "__main__":
    # Example 1: Extract from video file
    extract_frames_from_video(
        video_path="cctv_corrosion_footage.mp4",
        output_dir="training_frames",
        frame_interval=60  # 1 frame every 2 seconds at 30fps
    )
    
    # Example 2: Extract from RTSP stream
    # Uncomment and modify with your camera details:
    # extract_frames_from_rtsp(
    #     rtsp_url="rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101",
    #     output_dir="rtsp_training_frames",
    #     duration_seconds=600,  # 10 minutes
    #     frame_interval=90  # 1 frame every 3 seconds
    # )