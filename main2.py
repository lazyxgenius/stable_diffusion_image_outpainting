import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import replicate
import requests


def create_checkerboard_with_image(image_path, padding):
    """
    Create a checkerboard canvas and place the original image on it.

    Args:
        image_path (str): The path to the original image.
        padding (int): The amount of padding around the image.

    Returns:
        np.array: The new image with the original image placed on the checkerboard canvas.
    """
    # Load the original image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Read image with alpha channel if it exists
    # If the original image has 3 channels (RGB), add an alpha channel to make it RGBA
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image_height, image_width = image.shape[:2]

    # Calculate the canvas size with padding
    canvas_size = max(image_width, image_height) + 2 * padding

    check_size = 20  # Adjust this to your desired checkerboard square size

    # Adjust canvas size to be a multiple of check_size
    canvas_size = ((canvas_size + check_size - 1) // check_size) * check_size

    # Create a checkerboard canvas with transparency
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)  # Adding alpha channel
    num_checks = canvas_size // check_size

    for i in range(num_checks):
        for j in range(num_checks):
            if (i + j) % 2 == 0:
                canvas[i * check_size:(i + 1) * check_size, j * check_size:(j + 1) * check_size] = (255, 255, 255, 255)
            else:
                canvas[i * check_size:(i + 1) * check_size, j * check_size:(j + 1) * check_size] = (200, 200, 200, 255)

    # Calculate the position to place the original image on the canvas
    y_offset = (canvas_size - image_height) // 2
    x_offset = (canvas_size - image_width) // 2

    # Place the original image on the canvas
    canvas[y_offset:y_offset + image_height, x_offset:x_offset + image_width, :] = image

    return image, canvas, (x_offset, y_offset, image_width, image_height)


def create_mask(canvas_size, image_position):
    """
    Create a mask image.

    Args:
        canvas_size (int): The size of the checkerboard canvas.
        image_position (tuple): The position and size of the original image on the canvas.
        border (int): The border to shrink the black rectangle inside the original image.

    Returns:
        np.array: The mask image.
    """
    mask = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255  # Start with a white mask

    x_offset, y_offset, image_width, image_height = image_position

    border = 20  # Adjust this to your desired border size to shrink the black rectangle

    # Calculate the position of the slightly smaller black rectangle
    x1 = x_offset + border
    y1 = y_offset + border
    x2 = x_offset + image_width - border
    y2 = y_offset + image_height - border

    # Draw the black rectangle on the mask
    mask[y1:y2, x1:x2] = 0

    return mask


# Example usage
image_path = r"C:\Users\Aditya PC\PycharmProjects\image_outpainting\avocado_armchair.jpg"
padding = 500  # Adjust this to your desired padding


original_image, new_image_with_canvas, image_position = create_checkerboard_with_image(image_path, padding)
mask = create_mask(new_image_with_canvas.shape[0], image_position)

# Save the images
output_dir = os.path.dirname(image_path)
new_image_with_canvas_path = os.path.join(output_dir, "checkerboard_with_image.jpg")
mask_image_path = os.path.join(output_dir, "mask_image.jpg")

cv2.imwrite(new_image_with_canvas_path, new_image_with_canvas)
cv2.imwrite(mask_image_path, mask)




# Paths to your images
source_image_path = r"C:\Users\Aditya PC\PycharmProjects\image_outpainting\checkerboard_with_image.jpg"
mask_image_path = r"C:\Users\Aditya PC\PycharmProjects\image_outpainting\mask_image.jpg"

# Set your Replicate API token
api_token = "your_replicate_API_token"
# Set up the Replicate API client with authentication
os.environ["REPLICATE_API_TOKEN"] = api_token

# Set up the Replicate API client
output = replicate.run(
    "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
    input={
        "mask": open(r"C:\Users\Aditya PC\PycharmProjects\image_outpainting\mask_image.jpg", "rb"),
        "image": open(r"C:\Users\Aditya PC\PycharmProjects\image_outpainting\checkerboard_with_image.jpg", "rb"),
        "width": 512,
        "height": 512,
        "prompt": "a white room full of plants",
        "scheduler": "DPMSolverMultistep",
        "num_outputs": 1,
        "guidance_scale": 7.5,
        "num_inference_steps": 25
    }
)

# Print the output URL
print("Generated outpainted image URL:", output)

# Save the output image

output_url = output[0]  # Extract the URL from the list
response = requests.get(output_url)
outpainted_image_path = os.path.join(output_dir, "outpainted_image.jpg")
with open(outpainted_image_path, 'wb') as f:
    f.write(response.content)

# Display the original image, the new image with the checkerboard canvas, and the mask
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# New Image with Checkerboard Canvas
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(new_image_with_canvas, cv2.COLOR_BGR2RGB))
plt.title("New Image with Checkerboard Canvas")
plt.axis('off')

# Mask Image
plt.subplot(2, 2, 3)
plt.imshow(mask, cmap='gray')
plt.title("Mask Image")
plt.axis('off')

# Final image
outpainted_image = cv2.imread(outpainted_image_path)
outpainted_image_rgb = cv2.cvtColor(outpainted_image, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 4)
plt.imshow(outpainted_image_rgb)
plt.title("Outpainted Image")
plt.axis('off')

plt.show()



