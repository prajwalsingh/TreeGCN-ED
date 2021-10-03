import pymesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
np.random.seed(45)

# Refrences
# https://medium.com/@daviddelaiglesiacastro/3d-point-cloud-generation-from-3d-triangular-mesh-bbb602ecf238
# https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
# https://www.youtube.com/watch?v=HYAgJN3x4GA

def find_triangle_area(v1, v2, v3):
	'''
		This method is use to calculate area of triangle in 3d using
		cross product.
		Args:
			v1 : N x 3 (vertex 1 coordinates)
			v2 : N x 3 (vertex 2 coordinates)
			v3 : N x 1 (vertex 3 coordinates)
			traingle -> [v1 v2 v3]
	'''
	vec_yx = (v1-v2) # vector yx
	vec_yz = (v3-v2) # vector yz
	area_of_traingle = (1.0/2.0) * np.linalg.norm(np.cross(vec_yx, vec_yz), axis=1)
	return area_of_traingle

def mesh_to_pc(vertices, faces, N):
	'''
		This method is used to convert mesh into point cloud.
		Args:
			vectices : N x 3
			faces    : N x 3
			N        : number of points in point cloud
	'''
	vertices        = np.array(vertices)
	faces           = np.array(faces)
	triangle_v1_xyz = vertices[faces[:, 0]]
	triangle_v2_xyz = vertices[faces[:, 1]]
	triangle_v3_xyz = vertices[faces[:, 2]]

	area_of_traingles     = find_triangle_area(triangle_v1_xyz, triangle_v2_xyz, triangle_v3_xyz)
	probabilities         = area_of_traingles/np.sum(area_of_traingles)
	weighted_random_point = np.random.choice(range(area_of_traingles.shape[0]), size=N, p=probabilities)

	triangle_v1_xyz = triangle_v1_xyz[weighted_random_point]
	triangle_v2_xyz = triangle_v2_xyz[weighted_random_point]
	triangle_v3_xyz = triangle_v3_xyz[weighted_random_point]

	w1          = np.random.rand(N, 1)
	w2          = np.random.rand(N, 1)
	outside     = (w1 + w2) > 1
	w1[outside] = 1 - w1[outside]
	w2[outside] = 1 - w2[outside]


	P = triangle_v2_xyz + w1 * (triangle_v1_xyz - triangle_v2_xyz) + w2 * (triangle_v3_xyz - triangle_v2_xyz)
	P = (P - np.min(P, axis=0)) / (np.max(P, axis=0) - np.min(P, axis=0))
	return P

def genereate_point_cloud(mesh_path, N):
	'''
		This method is use to generate point cloud from given mesh
		Args:
			mesh_path : Path to mesh file
			N         : Number of points present in point cloud
	'''
	mesh  = pymesh.load_mesh(mesh_path)
	P     = mesh_to_pc(mesh.vertices, mesh.faces, N)
	return P

if __name__ == '__main__':
	P     = genereate_point_cloud(mesh_path='chair_0022.off', N=2000)
	# P     = genereate_point_cloud(mesh_path='bathtub_0001.off', N=2000)
	ax = plt.axes(projection='3d')
	ax.scatter3D(P[:, 0], P[:, 1], P[:, 2])
	plt.show()