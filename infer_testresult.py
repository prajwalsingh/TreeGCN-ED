import os
from glob import glob
from natsort import natsorted

def parse_data(data):
	data = list(map(str.strip, data))
	data = [float(item.split(':')[1]) for item in data]
	return float(sum(data))/float(len(data))

if __name__ == '__main__':
	files = natsorted(glob('test_results/*.txt'))
	print('Mean CD Per Class:\n')
	total_loss = 0.0
	for file_name in files:
		with open(file_name, 'r') as file:
			data = file.readlines()
		loss = parse_data(data)
		total_loss += loss
		print('{} : {}'.format( os.path.splitext(file_name.split(os.path.sep)[1])[0],  loss))

	print('Total loss: {}'.format(total_loss/float(len(files))))