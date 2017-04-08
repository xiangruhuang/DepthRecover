all: data/merged_100.npy
	((stdbuf -oL python main.py --batch_size=15) 2>&1) >> mylog

data/merged_100.npy:
	make -C ./data merged_100.npy
