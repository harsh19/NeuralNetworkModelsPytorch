import solver
import sys

import argparse

# Note embeddings_size is not used as of now. embeddings are made of same size as hidden size. TODO

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', dest='mode', default='train')
    parser.add_argument('--debug', dest='debug', action='store_true', default=False)
    parser.add_argument('--use_data_max_out', dest='use_data_max_out', action='store_true', default=False, help="use max output length as input length + 10")
    parser.add_argument('-run_name', dest='run_name', default='default')
    parser.add_argument('-cell_type', dest='cell_type', default='gru')
    parser.add_argument("-enc_hidden_size", type=int, dest="enc_hidden_size", help="enc_hidden_size", default=256)
    parser.add_argument("-dec_hidden_size", type=int, dest="dec_hidden_size", help="dec_hidden_size", default=256)
    parser.add_argument("-num_of_points", type=int, dest="num_of_points", help="num of training data points to use. -1 means all", default=-1)
    parser.add_argument("-enc_embedding_size", type=int, dest="enc_embedding_size", help="enc_embedding_size", default=256)
    parser.add_argument("-dec_embedding_size", type=int, dest="dec_embedding_size", help="dec_embedding_size", default=256)
    parser.add_argument("-epochs", type=int, dest="epochs", help="epochs", default=25)
    parser.add_argument("-batch_size", type=int, dest="batch_size", help="batch_size", default=32)
    parser.add_argument("-save_every_epoch", type=int, dest="save_every_epoch", help="save_every_epoch", default=1)
    parser.add_argument("-print_every_batch", type=int, dest="print_every_batch", help="print_every_batch", default=10)
    parser.add_argument("-max_length", type=int, dest="max_length", help="max_length", default=50)
    parser.add_argument("-max_out_length", type=int, dest="max_out_length", help="max_out_length", default=50)
    parser.add_argument("-teacher_forcing_ratio", type=float, dest="teacher_forcing_ratio", help="teacher_forcing_ratio", default=1.1)    
    parser.add_argument("-dec_dropout", type=float, dest="dec_dropout", help="dec_dropout. -1 is default. negative would be skipping dropout altogether", default=-1.0)    
    args = parser.parse_args()  
    return args

params = parseArguments()
if params.debug:
    params.num_of_points=96
    params.print_every_batch=1
for arg in vars(params):
    print arg + "=" + str( getattr(params, arg) )
seq2seq = solver.Solver(params)
seq2seq.main()
