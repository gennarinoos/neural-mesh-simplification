# Neural Mesh Simplification

This repository contains an implementation of the paper "Neural Mesh Simplification" by Potamias et al. (CVPR 2022). The project aims to provide a fast, learnable method for mesh simplification that generates simplified meshes in real-time.

Research, methodology introduced in the [Neural Mesh Simplification paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Potamias_Neural_Mesh_Simplification_CVPR_2022_paper.pdf), with the updated info shared in [supplementary material](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Potamias_Neural_Mesh_Simplification_CVPR_2022_supplemental.pdf).

This implementation could not have been done without the use of an LLM, specifically Claude Sonnet 3.5 by Anthropic. It was useful to create a project, upload the PDF of the papers there and use the custom instructions in [.cursorrules](.cursorrules). To steer the model, a copy of the file structure (which it helped create early on) is also useful. This can be created with the command `tree -F -I '*.ply|*.obj|*__pycache__*' > file_structure.txt` in the root directory of the project.

It is also useful to keep an updated copy of the main components of the code-base in the LLM project. This can be done with the following command, and uploading the file to the project in Claude:

```bash
find losses metrics models tests utils trainer \
    \( -type d -name "*__pycache__*" -o -name ".DS_Store" \) -prune -o -type f -print \
    | while IFS= read -r filepath; do
        echo "====== $filepath start ======" >> combined_output.txt
        cat "$filepath" >> combined_output.txt
        echo "====== end of $filepath ======" >> combined_output.txt
    done
```

## Overview

Neural Mesh Simplification is a novel approach to reduce the resolution of 3D meshes while preserving their appearance. Unlike traditional simplification methods that collapse edges in a greedy iterative manner, this method simplifies a given mesh in one pass using deep learning techniques.

The method consists of three main steps:

1. Sampling a subset of input vertices using a sophisticated extension of random sampling.
2. Training a sparse attention network to propose candidate triangles based on the edge connectivity of sampled vertices.
3. Using a classification network to estimate the probability that a candidate triangle will be included in the final mesh.

## Features

- Fast and scalable mesh simplification
- One-pass simplification process
- Preservation of mesh appearance
- Lightweight and differentiable implementation
- Suitable for integration into learnable pipelines

## Installation

```bash
git clone https://github.com/martinnormark/neural-mesh-simplification.git
cd neural-mesh-simplification
pip install -r requirements.txt
```

## Usage

```python
from neural_mesh_simplifier import NeuralMeshSimplifier

# Initialize the simplifier
simplifier = NeuralMeshSimplifier()

# Load a mesh
original_mesh = load_mesh("path/to/your/mesh.obj")

# Simplify the mesh
simplified_mesh = simplifier.simplify(original_mesh, target_faces=1000)

# Save the simplified mesh
save_mesh(simplified_mesh, "path/to/simplified_mesh.obj")
```

## Training

To train the model on your own dataset:

```bash
python ./scripts/train.py --data_dir /path/to/your/dataset --config /path/to/config.yaml --checkpoint_dir /path/to/checkpoints
```

To use default paths and config, you can run:

```base
uv run python3 scripts/train.py --data_dir data/processed/ --config configs/default.yaml
```

## Evaluation

To evaluate the model on a test set:

```bash
python ./scripts/evaluate.py --data_dir /path/to/test/set --config /path/to/config.yaml --checkpoint /path/to/checkpoints/best_model.pth
```

## Trainer Class

The `Trainer` class is responsible for managing the training and evaluation process. It includes methods for training, validation, checkpointing, logging, early stopping, learning rate scheduling, state management, error handling, and configuration management.

### Methods

- `__init__(self, config: Dict[str, Any])`: Initializes the trainer with the given configuration.
- `train(self)`: Trains the model for the specified number of epochs.
- `_train_one_epoch(self, epoch: int)`: Trains the model for one epoch.
- `_validate(self, epoch: int) -> float`: Validates the model and returns the validation loss.
- `_save_checkpoint(self, epoch: int, val_loss: float)`: Saves a checkpoint of the model.
- `_early_stopping(self, val_loss: float) -> bool`: Checks if early stopping should be triggered.
- `load_checkpoint(self, checkpoint_path: str)`: Loads a checkpoint from the specified path.
- `log_metrics(self, metrics: Dict[str, float], epoch: int)`: Logs the specified metrics.
- `evaluate(self, data_loader: DataLoader) -> Dict[str, float]`: Evaluates the model and returns the evaluation metrics.
- `handle_error(self, error: Exception)`: Handles errors that occur during training.
- `save_training_state(self, state_path: str)`: Saves the training state to the specified path.
- `load_training_state(self, state_path: str)`: Loads the training state from the specified path.

## Citation

If you use this code in your research, please cite the original paper:

```
@InProceedings{Potamias_2022_CVPR,
    author    = {Potamias, Rolandos Alexandros and Ploumpis, Stylianos and Zafeiriou, Stefanos},
    title     = {Neural Mesh Simplification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {18583-18592}
}
```

## Contributing

Contributions are welcome to improve this implementation. Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
