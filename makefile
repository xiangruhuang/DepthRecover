all: mnist.pkl
	python cnn_mnist.py

mnist.pkl: mnist.pkl.gz
	gunzip -k $<
