# Violence Detection Using Swin Transformer

This project leverages a Video Swin Transformer model to detect violence in videos. Follow the steps below to set up and run the project.

## Prerequisites

- Python 3.8 or later
- PyTorch 1.9 or later
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sadik-abd/violence_detection.git
    cd violence_detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Download Pre-trained Weights

1. Download the pre-trained model weights:
    [violence_swin.pth](https://github.com/sadik-abd/violence_detection/releases/download/v1/violence_swin.pth)

2. Place the downloaded file under the `weights/` directory.
    ```bash
    mkdir -p weights
    mv /path/to/violence_swin.pth weights/
    ```

## Usage

1. Specify the path to your video file in the `video_path` variable in the script.
    ```python
    # Example usage:
    video_path = "your_video_path_here.mp4"
    ```

2. Run the detection script:
    ```bash
    python test.py
    ```


## Contributing

Feel free to submit issues or pull requests if you find bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

**Note:** Ensure you have appropriate permissions for the video files you use and respect privacy and copyright regulations.

