# TreeGCN-ED
An autoencoder for point cloud encoding-decoding build on the work of tree-GAN

# Dataset Generation Step
* ShapeNetBenchmarkV2 dataset is used.
* To sample pointcloud from mesh:
  * https://medium.com/@daviddelaiglesiacastro/3d-point-cloud-generation-from-3d-triangular-mesh-bbb602ecf238
  * Area of triangle with 3 coordinates: https://math.stackexchange.com/a/1951650
  * https://www.youtube.com/watch?v=HYAgJN3x4GA
* Sample point cloud is spherical normalized.
* Data generation code is present in Preprocessing_Data folder.
* ShapeNetBenchmarkV2 numpy format dataset: [Link](https://drive.google.com/file/d/19aEXb_zVc99KG2qG0O23XZ9Z1sMyCCVw/view?usp=sharing)

# Pre-trained model
* Download pre-trained model from google drive:
  * [treeED_ckpt](https://drive.google.com/drive/folders/1BB39jONorejKfLeO4UQX84t3OlpSowQ0?usp=sharing)
  * [treeED_eckpt](https://drive.google.com/drive/folders/1IJy209nC8-V8ZvM55rhlnQ3iJZKIz2FF?usp=sharing)
* Keep treeED_ckpt, treeED_eckpt as it is in code directory.

# Results

<table style="width:100%; height:100%; border:none;">
       <tr>
             <td colspan=3 align="center">
                  <b>Intra-class interpolation results</b>
             </td>
       </tr>
       <tr>
             <td>
                 <img src="https://github.com/prajwalsingh/TreeGCN-ED/blob/main/results_gif/chair_to_chair_7.gif" style="width:310px; height:210px;"/>
             </td>
             <td>
                 <img src="https://github.com/prajwalsingh/TreeGCN-ED/blob/main/results_gif/table_to_table_4.gif" style="width:310px; height:210px;"/>
             </td>
             <td>
                 <img src="https://github.com/prajwalsingh/TreeGCN-ED/blob/main/results_gif/airplane_to_airplane_5.gif" style="width:310px; height:210px;"/>
             </td>
       </tr>
 <tr>
             <td colspan=3 align="center">
                  <b>Inter-class interpolation results</b>
             </td>
       </tr>
       <tr>
             <td>
                 <img src="https://github.com/prajwalsingh/TreeGCN-ED/blob/main/results_gif/10.laptop_to_plane.gif" style="width:310px; height:210px;"/>
             </td>
             <td>
                 <img src="https://github.com/prajwalsingh/TreeGCN-ED/blob/main/results_gif/14.mug_to_table.gif" style="width:310px; height:210px;"/>
             </td>
             <td>
                 <img src="https://github.com/prajwalsingh/TreeGCN-ED/blob/main/results_gif/16.car_to_chair.gif" style="width:310px; height:210px;"/>
             </td>
       </tr>
</table>

# Reference
[1] [3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions](https://arxiv.org/abs/1905.06292) [ Dong Wook Shu, Sung Woo Park, Junseok Kwon ]
