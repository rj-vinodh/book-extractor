import gradio as gr
import cv2
import torch
import os
from PIL import Image
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")
#model = torch.hub.load('ultralytics/yolov', 'yolov5s', pretrained=True)

def extract_books(video_path, log_fn, log_history):
    book_images = []
    if not video_path or not os.path.exists(video_path):
        log_fn("Error: Video file missing or invalid path.")
        return [], "No video provided or file not found."

    log_fn(f"Processing video: {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = vidcap.read()
    while success and frame_count < 100:
        if frame_count % 10 == 0:
            frame_path = f"frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            log_fn(f"Analyzing frame {frame_count}...")

            results = model(frame_path)
            log_fn(f"Detections: {results}")
            log_fn(f"Number of results: {len(results)}")
            log_fn(f"Results details: {results}")
            #boxes = results[0].boxes
            for i, result in enumerate(results):
                num_boxes = int(result.boxes.xyxy.shape[0]) if hasattr(result.boxes, "xyxy") else 0
                log_fn(f"Frame {frame_count} - number of boxes: {num_boxes}")

                if num_boxes == 0:
                    continue

                for b_idx in range(num_boxes):
                    xy = result.boxes.xyxy[b_idx]
                    x1 = int(xy[0].item())
                    y1 = int(xy[1].item())
                    x2 = int(xy[2].item())
                    y2 = int(xy[3].item())

                    cls = int(result.boxes.cls[b_idx].item())
                    conf = float(result.boxes.conf[b_idx].item())

                # x1, y1, x2, y2= result.boxes.xyxy[0]
                # cls = result.boxes.cls[0]
                # conf = result.boxes.conf[0]
                    if int(cls) == 73:
                        crop_coords = (x1, y1 , x2, y2)
                        # Crop the image
                        cropped = Image.open(frame_path).crop(crop_coords)
                        #cropped = Image.open(frame_path).crop((x1, y1, x2, y2))
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
    return book_images, "\n".join(log_history)

def main():
    log_history = []

    def log_fn(msg):
        log_history.append(msg)

    default_video = "./test.mp4"  # Adjust path as needed

    with gr.Blocks() as demo:
        gr.Markdown("# Book Extraction from Video with Logs")

        video_input = gr.Video(value=default_video, label="Upload or Use Default Video")
        process_button = gr.Button("Process Video")
        gallery = gr.Gallery(label="Extracted Book Images")
        logs_output = gr.Textbox(label="Logs", lines=10)

        def clear_logs():
            log_history.clear()

        def wrapped_process(video_path):
            clear_logs()
            images, logs = extract_books(video_path, log_fn, log_history)
            return images, logs

        process_button.click(
            fn=wrapped_process,
            inputs=video_input,
            outputs=[gallery, logs_output],
        )

    demo.launch()

if __name__ == "__main__":
    main()
