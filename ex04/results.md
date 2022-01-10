# python run_resnet.py --clean_eval
Loss 0.939719 accuracy 76.25%

# evaluate without update_bn_params
python run_resnet.py --evaluate --num_workers 4

{'defocus_blur': [55.4140127388535, 46.6062898089172, 30.712579617834397], 'glass_blur': [55.931528662420384, 42.88415605095541, 20.48168789808917], 'motion_blur': [61.146496815286625, 47.63136942675159, 29.468550955414013], 'zoom_blur': [48.059315286624205, 37.818471337579616, 30.59315286624204], 'snow': [53.16480891719745, 26.572452229299365, 31.399283439490443], 'frost': [59.016719745222936, 39.73925159235669, 26.871019108280255], 'fog': [56.99641719745223, 48.586783439490446, 37.44028662420382], 'brightness': [73.19864649681529, 71.48686305732484, 68.15286624203821]}

# evaluate with update_bn_params
python run_resnet.py --corruption "frost" --severity 2 --apply_bn --num_bn_updates 100

frost severity 2
Updating BN params (num updates:10)
Validation complete. Loss 2.450538 accuracy 48.25%

frost severity 2
Updating BN params (num updates:20)
Validation complete. Loss 2.528284 accuracy 47.87%

frost severity 2
Updating BN params (num updates:50)
Validation complete. Loss 2.679041 accuracy 46.05%

frost severity 2
Updating BN params (num updates:100)
Validation complete. Loss 2.725370 accuracy 45.38%

# evaluate with update_bn_params for different batch sizes
python run_resnet.py --corruption "frost" --severity 2 --apply_bn --num_bn_updates 50 --batch_size 1

batch size 1: Loss 5.182195 accuracy 24.44%
batch size 4: Loss 2.989547 accuracy 43.10%
batch size 16: Loss 2.722301 accuracy 45.31%
batch size 64: Loss 2.681277 accuracy 46.08%