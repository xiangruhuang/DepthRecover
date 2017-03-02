all:
	python mnist.pkl

mnist.pkl: mnist.pkl.gz
	gunzip $<
