import os

import cv2
import gradio as gr
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection, pipeline


# Load the Hugging Face DETR checkpoint using safetensors weights.
MODEL_ID = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForObjectDetection.from_pretrained(
    MODEL_ID,
    use_safetensors=True,
)
detector = pipeline(
    "object-detection",
    model=model,
    image_processor=image_processor,
    device=0 if torch.cuda.is_available() else -1,
)


def extract_books(video_path, log_fn, log_history):
    book_images = []
    if not video_path or not os.path.exists(video_path):
        log_fn("Error: Video file missing or invalid path.")
        return [], "No video provided or file not found."

    log_fn(f"Processing video: {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        log_fn("Error: Could not open the video file.")
        return [], "Could not open the video file."

    frame_count = 0
    try:
        success, frame = vidcap.read()
        while success and frame_count < 100:
            if frame_count % 10 == 0:
                log_fn(f"Analyzing frame {frame_count}...")

                # OpenCV frames are BGR; the Hugging Face image pipeline expects RGB.
                frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detections = detector(frame_image)
                books_in_frame = 0

                for detection in detections:
                    label = detection["label"].lower()
                    confidence = float(detection["score"])
                    if label != "book" or confidence <= 0.5:
                        continue

                    box = detection["box"]
                    x1 = max(0, int(box["xmin"]))
                    y1 = max(0, int(box["ymin"]))
                    x2 = min(frame_image.width, int(box["xmax"]))
                    y2 = min(frame_image.height, int(box["ymax"]))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop_path = f"book_{frame_count}_{books_in_frame}.jpg"
                    frame_image.crop((x1, y1, x2, y2)).save(crop_path)
                    book_images.append(crop_path)
                    books_in_frame += 1
                    log_fn(
                        f"Book detected with confidence {confidence:.2f} "
                        f"at frame {frame_count}"
                    )

                log_fn(f"Frame {frame_count} - book detections: {books_in_frame}")

            success, frame = vidcap.read()
            frame_count += 1
    finally:
        vidcap.release()

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

        def wrapped_process(video_path):
            log_history.clear()
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
