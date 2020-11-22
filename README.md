# Recommended specification:
	python - 3.7.4
	numpy - 1.19.2
	cudatoolkit - 10.2.89
	cudnn - 7.6.5
	pytorch - 1.6.0

These codes are designed to run in a multi-GPU environment. If it is not available for you, please pay attention to the variable "device_id".

Please run each .py file separately in the project path.

Larger datasets like MNIST are not contained in the "datasets" folder. Please redirect routes inside the codes.

For experiment 3, please run "lab3.py" to collect results and then run "lab3_eval.py" to visualize them. All results can be found in the "results/" folder.
