python run_cifar.py --test_dataloader
python run_cifar.py --test_model
python run_cifar.py --out_dir=no_augment --num_epochs=256 --batch_size=256 --transforms=basic --optimizer=adamw
python run_cifar.py --out_dir=augment --num_epochs=256 --batch_size=256 --transforms=own --optimizer=adamw

ssh -L 16010:127.0.0.1:6010 username@login.informatik.uni-freiburg.de
tensorboard --logdir=. --port 6010
http://127.0.0.1:16010/