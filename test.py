import gradio as gr
import cv2
import torch
import os
from PIL import Image

# Load YOLOv5 model (make sure ultralytics installed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def extract_books(video_path, log_fn):
    book_images = []
    if not video_path or not os.path.exists(video_path):
        log_fn("Error: Video file missing or invalid path.")
        return [], "No video provided or file not found."

    log_fn(f"Processing video: {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = vidcap.read()
    while success and frame_count < 100:  # limit frames for faster demo
        if frame_count % 10 == 0:  # process every 10th frame for speed
            frame_path = f"frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            log_fn(f"Analyzing frame {frame_count}...")
            
            results = model(frame_path)
            boxes = results.xyxy[0]
            for i, box in enumerate(boxes):
                x1, y1, x2, y2, conf, cls = box
                if int(cls) == 73 and conf > 0.5:  # 73 is 'book'
                    cropped = Image.open(frame_path).crop((x1, y1, x2, y2))
                    crop_path = f"book_{frame_count}_{i}.jpg"
                    cropped.save(crop_path)
                    book_images.append(crop_path)
                    log_fn(f"Book detected with confidence {conf:.2f} at frame {frame_count}")
        success, frame = vidcap.read()
        frame_count += 1

    if not book_images:
        log_fn("No books detected in the video.")
    else:
        log_fn(f"Extraction complete. Found {len(book_images)} book images.")
    return book_images, "\n".join(log_fn.history)

def main():
    log_messages = []
    def log_fn(msg):
        log_messages.append(msg)
    
    default_video = "./test.mp4"  # Put a real path accessible inside container

    with gr.Blocks() as demo:
        gr.Markdown("# Book Extraction from Video with Logs")

        video_input = gr.Video(value=default_video, label="Upload or Use Default Video")
        process_button = gr.Button("Process Video")
        gallery = gr.Gallery(label="Extracted Book Images") #.style(grid=[4], height="auto")
        logs_output = gr.Textbox(label="Logs", lines=10)

        def clear_logs():
            log_messages.clear()

        def wrapped_process(video_path):
            clear_logs()
            images, logs = extract_books(video_path, log_fn)
            return images, logs

        process_button.click(
            fn=wrapped_process,
            inputs=video_input,
            outputs=[gallery, logs_output],
        )

    demo.launch()

if __name__ == "__main__":
    main()
