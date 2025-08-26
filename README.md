# OpenCV-Mouse-Controller
A Python application that uses OpenCV to control the mouse cursor with hand gestures captured via a webcam.
## Features
- Move the mouse cursor using hand movements.
- Click using specific hand gestures.
- Scroll using hand gestures.
## Requirements
- Python 3.x
- OpenCV
- NumPy
- PyAutoGUI
- Mediapipe
## Installation
1. Clone the repository:
```bash
git clone https://github.com/Muhamid786/OpenCV-Mouse-Controller.git
cd OpenCV-Mouse-Controller
```
2. Install the required packages:
```bash
pip install opencv-python numpy pyautogui mediapipe
```
## GPU Acceleration (Optional)
For better performance with NVIDIA GPUs:
1. Install NVIDIA drivers:
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```
and verify installation via `nvidia-smi`

## Usage
1. Run the application:
```bash
python mouse_controller.py
```
2. Follow the on-screen instructions to control the mouse using hand gestures.
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details
.
## Acknowledgements
- [OpenCV](https://opencv.org/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest)
- [Mediapipe](https://mediapipe.dev/)
- Inspired by various online tutorials and resources on hand gesture recognition and mouse control.
## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.
## Contact
For any questions or suggestions, please open an issue on the GitHub repository.