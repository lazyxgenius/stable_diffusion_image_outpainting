
# Stable Diffusion Outpainting

## Project Description

This project provides a web-based interface for generating outpainted images using the Stable Diffusion model. Users can upload an image, define padding, and provide a prompt to guide the outpainting process. The application leverages the Replicate API to generate the outpainted image, allowing for seamless integration and user interaction through an easy-to-use interface.

### Key Features

- **Image Upload**: Users can upload an image for outpainting.
- **Padding Specification**: Define the amount of padding to apply around the original image before generating the outpainted sections.
- **Prompt Input**: Provide a text prompt to guide the AI in generating the outpainted image.
- **Image Preview and Download**: After processing, the original, checkerboard, mask, and outpainted images are displayed, and users can download them directly from the interface.

### Technology Stack

- **Backend**: Flask - Python
- **Frontend**: HTML, CSS (Bootstrap), JavaScript
- **API Integration**: Replicate API for Stable Diffusion model


### Usage

1. Upload an image that you want to outpaint.
2. Specify the padding and provide a text prompt to guide the AI.
3. Click on "Generate Outpainting" to process the image.
4. Once the images are generated, they will be displayed on the page with download options.

### Reference
https://replicate.com/guides/stable-diffusion/outpainting#:~:text=Outpainting%20is%20the%20process%20of,a%20region%20outside%20of%20it.

