all: data/merged_100.npy
	python cnn_mnist.py

data/merged_100.npy:
	make -C ./data merged_100.npy
