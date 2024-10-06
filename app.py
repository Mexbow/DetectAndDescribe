from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import os
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

app = Flask(__name__)

# Load models once when the app starts
weights_path = 'weights/best14.pt'
object_detection_model = torch.hub.load('Mexbow/yolov5_model', 'custom', path=weights_path, autoshape=True)
captioning_processor = AutoImageProcessor.from_pretrained("motheecreator/ViT-GPT2-Image-Captioning")
tokenizer = AutoTokenizer.from_pretrained("motheecreator/ViT-GPT2-Image-Captioning")
caption_model = VisionEncoderDecoderModel.from_pretrained("motheecreator/ViT-GPT2-Image-Captioning")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    image_path = os.path.join('static', 'uploaded_image.jpg')
    file.save(image_path)
    
    img = Image.open(image_path).convert('RGB')
    results, original_caption = process_image(img)
    
    return render_template('results.html', results=zip(results['labels'], results['captions']), original_caption=original_caption)

def process_image(image):
    results = object_detection_model(image)
    
    img_with_boxes = results.render()[0]
    detected_image_path = os.path.join('static', 'detected_image.jpg')
    img_with_boxes = Image.fromarray(img_with_boxes)
    img_with_boxes.save(detected_image_path)
    
    boxes = results.xyxy[0][:, :4].cpu().numpy()
    labels = [results.names[int(x)] for x in results.xyxy[0][:, 5].cpu().numpy()]

    # Caption the original image
    original_inputs = captioning_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        caption_ids = caption_model.generate(**original_inputs)
    original_caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)

    # Crop objects and caption them
    cropped_images = crop_objects(image, boxes)
    captions = []
    for cropped_image in cropped_images:
        inputs = captioning_processor(images=cropped_image, return_tensors="pt")
        with torch.no_grad():
            caption_ids = caption_model.generate(**inputs)
        caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)
        captions.append(caption)
    
    return {'labels': labels, 'captions': captions, 'detected_image_path': detected_image_path}, original_caption


def crop_objects(image, boxes):
    cropped_images = []
    for box in boxes:
        cropped_image = image.crop((box[0], box[1], box[2], box[3]))
        cropped_images.append(cropped_image)
    return cropped_images

if __name__ == '__main__':
    app.run(debug=True)
