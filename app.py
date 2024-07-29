from flask import Flask, render_template, request, send_file
import os
import requests
import replicate
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def create_checkerboard_with_image(image_path, padding):
    print("Creating checkerboard with image")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # If the original image has 3 channels (RGB), add an alpha channel to make it RGBA
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image_height, image_width = image.shape[:2]

    canvas_size = max(image_width, image_height) + 2 * padding
    check_size = 20

    canvas_size = ((canvas_size + check_size - 1) // check_size) * check_size

    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    num_checks = canvas_size // check_size

    for i in range(num_checks):
        for j in range(num_checks):
            if (i + j) % 2 == 0:
                canvas[i * check_size:(i + 1) * check_size, j * check_size:(j + 1) * check_size] = (255, 255, 255, 255)
            else:
                canvas[i * check_size:(i + 1) * check_size, j * check_size:(j + 1) * check_size] = (200, 200, 200, 255)

    y_offset = (canvas_size - image_height) // 2
    x_offset = (canvas_size - image_width) // 2

    # Place the original image on the canvas
    canvas[y_offset:y_offset + image_height, x_offset:x_offset + image_width, :] = image

    return image, canvas, (x_offset, y_offset, image_width, image_height)


def create_mask(canvas_size, image_position):
    print("Creating mask")
    mask = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255

    x_offset, y_offset, image_width, image_height = image_position
    border = 20

    x1 = x_offset + border
    y1 = y_offset + border
    x2 = x_offset + image_width - border
    y2 = y_offset + image_height - border

    mask[y1:y2, x1:x2] = 0

    return mask


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            api_token = request.form['api_token']
            prompt = request.form['prompt']
            padding = int(request.form['padding'])
            image_file = request.files['image']

            image_path = os.path.join(UPLOAD_FOLDER, 'input_image.png')
            image_file.save(image_path)

            original_image, new_image_with_canvas, image_position = create_checkerboard_with_image(image_path, padding)
            mask = create_mask(new_image_with_canvas.shape[0], image_position)

            output_dir = os.path.dirname(image_path)
            original_image_path = os.path.join(output_dir, "original_image.png")
            checkerboard_with_image_path = os.path.join(output_dir, "checkerboard_with_image.png")
            mask_image_path = os.path.join(output_dir, "mask_image.png")

            cv2.imwrite(original_image_path, original_image)
            cv2.imwrite(checkerboard_with_image_path, new_image_with_canvas)
            cv2.imwrite(mask_image_path, mask)

            os.environ["REPLICATE_API_TOKEN"] = api_token

            output = replicate.run(
                "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                input={
                    "mask": open(mask_image_path, "rb"),
                    "image": open(checkerboard_with_image_path,
                                  "rb"),
                    "width": 512,
                    "height": 512,
                    "prompt": prompt,
                    "scheduler": "DPMSolverMultistep",
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 25
                }
            )
            print("Replicate API Output:", output)  # Debug statement

            # Process only the first image URL
            output_url = output[0]  # Ensure only the first image URL is used
            print("Output URL:", output_url)  # Debug statement

            response = requests.get(output_url)
            print("Response Status Code:", response.status_code)  # Debug statement
            print("Response Content Type:", response.headers['Content-Type'])  # Debug statement

            if response.status_code == 200:
                outpainted_image_path = os.path.join(output_dir, "outpainted_image.png")
                with open(outpainted_image_path, 'wb') as f:
                    f.write(response.content)
                print("Outpainted image saved at:", outpainted_image_path)  # Debug statement
            else:
                print("Failed to retrieve outpainted image:", response.text)  # Debug statement

            return render_template('index.html', original_image_path=original_image_path,
                                   checkerboard_with_image_path=checkerboard_with_image_path,
                                   mask_image_path=mask_image_path,
                                   outpainted_image_path=outpainted_image_path)
        except Exception as e:
            print(f"An error occurred: {e}")

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
