# Snake AI Reinforcement Learning

This repository contains two versions of a Snake game AI that uses Deep Q-Networks (DQN) to learn how to play the classic Snake game through reinforcement learning. Both versions support GPU acceleration on their respective platforms:

- **Mac Version (Apple Silicon with MPS support)**  
  Uses Apple's Metal Performance Shaders (MPS) to accelerate training if available. 

- **Windows Version (CUDA/CPU fallback)**  
  Designed for Windows environments. This version will attempt to use CUDA for GPU acceleration if available; otherwise, it falls back to CPU. 

## Features

- **Deep Q-Network (DQN):** Implements a three-layer neural network with dropout and weight decay to help prevent overfitting.
- **Headless Training Mode:** Disables the game window during training to maximize speed.
- **GPU Acceleration:** Supports Apple Silicon's MPS on Mac and CUDA on Windows (with a CPU fallback).
- **Real-Time Visualization:** Demo mode shows the AI playing Snake live.
- **Model Saving:** Automatically saves the model when a new high score is reached.

## Requirements

- Python 3.8 or later
- [PyTorch](https://pytorch.org) (Ensure you install the version that supports your platform's GPU acceleration)
- [Pygame](https://www.pygame.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/) (for training mode visualization on Mac)

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/kl-charizard/snake-ai-python
cd snake-ai-python
pip install torch torchvision torchaudio pygame numpy matplotlib
```


## Platform-Specific Notes

- **Mac Version (Apple Silicon):**  
  - Ensure you have the latest PyTorch version that supports MPS.

- **Windows Version:**  
  - The script `snake_ai_windows.py` will attempt to use CUDA for GPU acceleration. If CUDA is unavailable, it will automatically fall back to using the CPU.

## Usage

### Mac & Windows Version

#### Training Mode

Run the Windows script in training mode (headless):

```bash
python snake_ai.py
```

- The game runs in headless mode to speed up training.
- Training progress (game count, current score, record score) is displayed in the console.
- The model is saved to `model.pth` when new records are reached.

#### Demo Mode

To load the trained model and visualize the AI playing the game, run:

```bash
python snake_ai.py demo
```

- A Pygame window will open to display the AI in action.
- Make sure the `model.pth` file is available in the working directory.

## Contributing

Contributions, suggestions, and improvements are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the developers and communities behind PyTorch and Pygame for providing robust tools to build and experiment with reinforcement learning.
- This project is inspired by various online tutorials and the enthusiasm of the open-source community for AI and game development.


## C++ version

- Coming soon
