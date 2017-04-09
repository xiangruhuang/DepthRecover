batch_size=10
learning_rate=1e-3
kernel_sizes=5-7-9

all: data/merged_100.npy
	$(eval namesuffix := b$(batch_size)_lr$(learning_rate)_ks$(kernel_sizes))
	mkdir -p logs
	$(eval model_dir := vanilla/run_$(namesuffix))
	mkdir -p $(model_dir)
	cat script_template | sed "s/LOGNAME/vanilla\/run_$(namesuffix)\/log/" > $(model_dir)/script
	echo "((stdbuf -oL python main.py --model_dir=$(model_dir) --batch_size=$(batch_size) --learning_rate=$(learning_rate) --kernel_sizes=$(kernel_sizes)) 2>&1) > $(model_dir)/log" >> $(model_dir)/script
	sbatch -A CS395T-Advanced-Geom $(model_dir)/script
	#((stdbuf -oL python main.py --model_dir=$(model_dir) --batch_size=$(batch_size) --learning_rate=$(learning_rate) --kernel_sizes=$(kernel_sizes)) 2>&1) > $(model_dir)/log
	#python main.py --batch_size=$(batch_size) --learning_rate=$(learning_rate) --kernel_sizes=$(kernel_sizes)

data/merged_100.npy:
	make -C ./data merged_100.npy

null:=
space:= $(null) #
comma:= ,
data=omnetpp
port=10888

plot_tensorboard:
	tensorboard --logdir=./tensorboard --port=$(port)

%.tensorboard:
	$(eval names := $(basename $@))
	$(eval names := $(subst +, ,$(names)))
	$(eval LOGDIR := $(foreach name,$(names),$(name):$(data)/$(name)/tensorboard))
	$(eval LOGDIR := $(subst $(space),$(comma),$(LOGDIR)))
	tensorboard --logdir=$(LOGDIR) --port=$(port)
