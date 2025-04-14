import torch
import pdb
import argparse

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description='Rationale-Net Classifier')
    #setup
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
    # learning
    parser.add_argument('--init_lr', type=float, default=2e-5, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs for train [default: 256]')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('--patience', type=int, default=999, help='Num epochs of no dev progress before half learning rate [default: 10]')
    #paths
    parser.add_argument('--save_dir', type=str, default='saved', help='where to save the snapshot')
    parser.add_argument('--results_dir', type=str, default='results', help='where to dump model config and epoch stats. If get_rationales is set to true, rationales for the test set will also be stored here.')
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    # model
    parser.add_argument('--model_form', type=str, default='bert', help='model is cnn/bilstm/bert' )
    parser.add_argument('--hidden_dim', type=int, default=256, help="Dim of hidden layer")
    parser.add_argument('--num_layers', type=int, default=2, help="Num layers of model_form to use")
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.3]')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='L2 norm penalty [default: 1e-3]')
    parser.add_argument('--filter_num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--filters', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    # bilstm
    parser.add_argument('--bilstm_dim', type=int, default=256, help='number of lstm nodes')
    # data
    parser.add_argument('--dataset', default='imdb', help='choose which dataset to run on. [default: imdb]')

    # gumbel
    parser.add_argument('--gumbel_temprature', type=float, default=1, help="Start temprature for gumbel softmax. This is annealed via exponential decay")
    parser.add_argument('--gumbel_decay', type=float, default=1e-5, help="Start temprature for gumbel softmax. This is annealed via linear decay")
    # rationale
    parser.add_argument('--get_rationales',  action='store_true', default=False, help="otherwise, just train encoder")
    parser.add_argument('--select_lambda', type=float, default=.01, help="y1 in Gen cost L + y1||z|| + y2|zt - zt-1| + y3|{z}|")
    parser.add_argument('--contig_lambda', type=float, default=0, help="y2 in Gen cost L + y1||z|| + y2|zt - zt-1|+ y3|{z}|")
    
    #experiments
    parser.add_argument('--rand_seed', type=int, default=2021, help="Random seed for torch")
    parser.add_argument('--warmup',  action='store_true', default=False, help="Linear LR warm up")
    parser.add_argument('--datasize', type=int, default=5000, help="The number of examples used for training")
    parser.add_argument('--kd',  action='store_true', default=False, help="If do knowledge distillation")
    parser.add_argument('--student_form', type=str, default='cnn', help='student model is cnn/bilstm/bert' )
    parser.add_argument('--kd_lambda', type=float, default=0, help="lambda for kd loss")
    
    
    args = parser.parse_args()
    # update args and print
    args.filters = [int(k) for k in args.filters.split(',')]

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


