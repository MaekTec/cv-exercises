python eval.py --output output --model FlowNetC --dataset FlyingThings3D --restore /project/cv-ws2122/shared-data1/OpticalFlowPretrainedModels/flownet_c/checkpoint-model-iter-000600000.pt
aepe    29.11129

python train.py --output output --model FlowNetS --dataset Sintel --restore /project/cv-ws2122/shared-data1/OpticalFlowPretrainedModels/flownet_s/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 610000

python train.py --output output --model FlowNetS --dataset Sintel --restore /project/cv-ws2122/shared-data1/OpticalFlowPretrainedModels/flownet_s/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 610000 --photometric --smoothness_loss