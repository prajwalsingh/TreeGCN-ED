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

<!-- # Results
  <table style="width:100%; height:100%; border:none;">
          <tr>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65010.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65030.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65259.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65279.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65508.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65528.png" style="width:128px; height:128px;"/>
               </td>
          </tr>
          <tr>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65757.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65777.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/66006.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/66026.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/66255.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/66275.png" style="width:128px; height:128px;"/>
               </td>
          </tr>
  </table>
 -->
# Reference
[1] [3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions](https://arxiv.org/abs/1905.06292) [ Dong Wook Shu, Sung Woo Park, Junseok Kwon ]
