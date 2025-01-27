# Mesh AI Assist

A collection of AI tools to work with 3D Meshes.

1. Neural Mesh Simplification

---

## 1. Neural Mesh Simplification

Implementation of the
paper [Neural Mesh Simplification paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Potamias_Neural_Mesh_Simplification_CVPR_2022_paper.pdf)
by Potamias et al. (CVPR 2022) and the updated info shared
in [supplementary material](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Potamias_Neural_Mesh_Simplification_CVPR_2022_supplemental.pdf).

This Python package provides a fast, learnable method for mesh simplification that generates simplified meshes in
real-time.

### Overview

Neural Mesh Simplification is a novel approach to reduce the resolution of 3D meshes while preserving their appearance.
Unlike traditional simplification methods that collapse edges in a greedy iterative manner, this method simplifies a
given mesh in one pass using deep learning techniques.

The method consists of three main steps:

1. Sampling a subset of input vertices using a sophisticated extension of random sampling.
2. Training a sparse attention network to propose candidate triangles based on the edge connectivity of sampled
   vertices.
3. Using a classification network to estimate the probability that a candidate triangle will be included in the final
   mesh.

### Features

- Fast and scalable mesh simplification
- One-pass simplification process
- Preservation of mesh appearance
- Lightweight and differentiable implementation
- Suitable for integration into learnable pipelines

### Installation

```bash
conda create -n neural-mesh-simplification python=3.12
conda activate neural-mesh-simplification
conda install pip
pip install -r requirements.txt
pip install -e .
```

### Example Usage / Playground

1. Drop your meshes as `.obj` files to the `examples/data` folder
2. Run the following command

```bash
python examples/example.py
```

3. Collect the simplified meshes in `examples/data`. The simplified mesh objects file name will be the ones prefixed
   with `simplified_`.

### Data Preparation

If you don't have a dataset for training and evaluation, you can use a collection from HuggingFace's 3D Meshes dataset.
See https://huggingface.co/datasets/perler/ppsurf for more information.

Run the following script to use the HuggingFace API to download

```bash
python scripts/download_test_meshes.py
```

Data will be downloaded in the `data/raw` folder at the root of the project.
You can use `--target-folder` to specify a different folder.

Once you have some data, you should preprocess it using the following script:

```bash
python scripts/preprocess.py
```

You can use the `--data_path` argument to specify the path to the dataset. The script will create a `data/processed`

### Training

To train the model on your own dataset with the prepared data:

```bash
python ./scripts/train.py --data_path data/processed --config configs/default.yaml
```

Specify a different `--data_path` if you have your data in a different location.
You can use the default training config at `scripts/train_config.yml` or specify a different one with `--config_path`.
You can also override the checkpoint directory specified in the config file (where the model will be saved) with
`--checkpoint_dir`.
If the training was interrupted, you can resume it by specifying the path to the checkpoint with `--resume`.

### Evaluation

To evaluate the model on a test set:

```bash
python ./scripts/evaluate.py --config configs/default.yaml --eval_data_path /path/to/test/set --checkpoint /path/to/checkpoint.pth
```

### Inference

To simplify a mesh using the trained model:

```bash
python ./scripts/inference.py --input-file /path/to/your/mesh.obj --model-checkpoint /path/to/checkpoint.pth --device cpu
```

This will create a file next to the input file with suffix `_simplified`
If you have a CUDA-compatible GPU, you can specify `--device cuda` to use it for inference.

### Citation

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
