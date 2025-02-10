# Snake AI Reinforcement Learning

This repository contains two versions of a Snake game AI that uses Deep Q-Networks (DQN) to learn how to play the classic Snake game through reinforcement learning. Both versions support GPU acceleration on their respective platforms:

- **Mac Version (Apple Silicon with MPS support)**  
  Uses Apple's Metal Performance Shaders (MPS) to accelerate training if available. This version is split into multiple files:
  - `snake_ai_core.py`: Contains the core game environment, neural network, training agent, and utilities.
  - `train_snake_ai.py`: Runs training in headless mode (no game window) for faster performance while visualizing score progress using Matplotlib.
  - `demo_snake_ai.py`: Loads a saved model and visually demonstrates the AI playing Snake in a Pygame window.

- **Windows Version (CUDA/CPU fallback)**  
  Designed for Windows environments. This version will attempt to use CUDA for GPU acceleration if available; otherwise, it falls back to CPU. The complete implementation is provided in:
  - `snake_ai_windows.py`: A single-file script that supports both training (headless mode) and demo mode (graphical display) based on command-line arguments.

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
git clone https://github.com/yourusername/snake-ai.git
cd snake-ai
pip install torch torchvision torchaudio pygame numpy matplotlib
