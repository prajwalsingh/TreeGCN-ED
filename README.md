### TreeGCN-ED
An autoencoder for point cloud encoding-decoding build using tree-GAN as base work.

[[Paper Link](https://arxiv.org/abs/2110.03170)]

### Dataset Generation Step
* ShapeNetBenchmarkV2 dataset is used.
* To sample pointcloud from mesh:
  * https://medium.com/@daviddelaiglesiacastro/3d-point-cloud-generation-from-3d-triangular-mesh-bbb602ecf238
  * Area of triangle with 3 coordinates: https://math.stackexchange.com/a/1951650
  * https://www.youtube.com/watch?v=HYAgJN3x4GA
* Sample point cloud is spherical normalized.
* Data generation code is present in Preprocessing_Data folder.
* ShapeNetBenchmarkV2 numpy format dataset: [Link](https://iitgnacin-my.sharepoint.com/:u:/g/personal/singh_prajwal_iitgn_ac_in/EUVQLfQ7J1FUHi9UfzkeDksBoBRhKcB4F-djsXU986dMQw?e=Tr51es)

### Pre-trained model
* Download pre-trained model from google drive:
  * [treeED_ckpt](https://iitgnacin-my.sharepoint.com/:f:/g/personal/singh_prajwal_iitgn_ac_in/EvrEyXkgNihMo83Yl4QrFZMBeJg3Ow-MqichqIbzq2UCKw?e=jmF0Ee)
  * [treeED_eckpt](https://iitgnacin-my.sharepoint.com/:f:/g/personal/singh_prajwal_iitgn_ac_in/Et25Augmb35Kp-3Gn3dcCH4B5D7w9cr2yrY5dbLLuI2rPw?e=A8vAuY)
* Keep treeED_ckpt, treeED_eckpt as it is in code directory.

### Results

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
          <td  align="center">
               Chair to Chair
          </td>
          <td  align="center">
               Table to Table
          </td>
          <td  align="center">
               Airplane to Airplane
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
       <tr>
          <td  align="center">
               Laptop to Airplane
          </td>
          <td  align="center">
               Cup to Table
          </td>
          <td  align="center">
               Car to Chair
          </td>
      </tr>
</table>

### Reference
[1] [3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions](https://arxiv.org/abs/1905.06292) [ Dong Wook Shu, Sung Woo Park, Junseok Kwon ]

### Bibtex Citation

<pre>
@misc{singh2021treegcned,
      title={TreeGCN-ED: Encoding Point Cloud using a Tree-Structured Graph Network}, 
      author={Prajwal Singh and Kaustubh Sadekar and Shanmuganathan Raman},
      year={2021},
      eprint={2110.03170},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
</pre>
