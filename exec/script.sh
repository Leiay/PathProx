
# MNIST subset + MLP factorized
n_gpu=0
wd=0.0001
lr=0.3
config_name='./configs/mnist_subset_mlp.yml'
logger_name="pathprox"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --wd-param $wd --with-prox-upd --lr $lr

logger_name="wd"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --wd-param $wd --with-loss-term --lr $lr


# MNIST + MLP
wd=0.0001
lr=0.1
config_name='./configs/mnist_mlp.yml'
logger_name="pathprox"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --wd-param $wd --with-prox-upd --lr $lr

logger_name="wd"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --wd-param $wd --with-loss-term --lr $lr


# CIFAR10 + VGG19
n_gpu=1
wd=0.001
lr=0.1
config_name='./configs/cifar_vgg.yml'
logger_name="pathprox"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --wd-param $wd --with-prox-upd --lr $lr

logger_name="wd"
python main.py --cuda --gpu $n_gpu --logger-name $logger_name --config $config_name --wd-param $wd --with-loss-term --lr $lr

