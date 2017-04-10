batch_size=5
learning_rate=1e-3
kernel_sizes=3-3-3
gaussian_sizes=11-11
conv_size=7
target_frac=0.01
mask_rate=0.01
lambda_1=1.0
lambda_2=1.0
lambda_3=1.0

all: data/merged_100.npy
	$(eval namesuffix := b$(batch_size)_lr$(learning_rate)_ks$(kernel_sizes)_mr$(mask_rate)_tf$(target_frac))
	mkdir -p logs
	$(eval model_dir := vanilla/run_$(namesuffix))
	mkdir -p $(model_dir)
	cat script_template | sed "s/LOGNAME/vanilla\/run_$(namesuffix)\/log/" > $(model_dir)/script
	$(eval params := --model_dir=$(model_dir) --batch_size=$(batch_size) --learning_rate=$(learning_rate) --kernel_sizes=$(kernel_sizes) --gaussian_sizes=$(gaussian_sizes) --target_frac=$(target_frac) --conv_size=$(conv_size) --mask_rate=$(mask_rate) --lambda_1=$(lambda_1) --lambda_2=$(lambda_2) --lambda_3=$(lambda_3) )
	echo "((stdbuf -oL python main.py $(params)) 2>&1) > $(model_dir)/log" >> $(model_dir)/script
	sbatch -A CS395T-Advanced-Geom $(model_dir)/script
	#((stdbuf -oL python main.py $(params)) 2>&1) > $(model_dir)/log
	#python main.py $(params)

data/merged_100.npy:
	make -C ./data merged_100.npy

null:=
space:= $(null) #
comma:= ,
data=vanilla
port=10888

plot_tensorboard:
	tensorboard --logdir=./tensorboard --port=$(port)

%.tensorboard:
	$(eval names := $(basename $@))
	$(eval names := $(subst +, ,$(names)))
	$(eval LOGDIR := $(foreach name,$(names),$(name):$(data)/$(name)/tensorboard))
	$(eval LOGDIR := $(subst $(space),$(comma),$(LOGDIR)))
	tensorboard --logdir=$(LOGDIR) --port=$(port)

folder=vanilla
boards_under:
	tensorboard --logdir=$(folder) --port=$(port)
