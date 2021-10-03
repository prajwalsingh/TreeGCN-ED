# TreeGCN-ED
An autoencoder for point cloud encoding-decoding build on the work of tree-GAN

# Reference
[1] [3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions](https://arxiv.org/abs/1905.06292) [ Dong Wook Shu, Sung Woo Park, Junseok Kwon ]

# Dataset Generation Step
* ShapeNetBenchmarkV2 dataset is used.
* To sample pointcloud from mesh:
  * https://medium.com/@daviddelaiglesiacastro/3d-point-cloud-generation-from-3d-triangular-mesh-bbb602ecf238
  * Area of triangle with 3 coordinates: https://math.stackexchange.com/a/1951650
  * https://www.youtube.com/watch?v=HYAgJN3x4GA
* Data generation code is present in Preprocessing_Data folder.
* ShapeNetBenchmarkV2 numpy format dataset: [Link](https://drive.google.com/file/d/19aEXb_zVc99KG2qG0O23XZ9Z1sMyCCVw/view?usp=sharing)
