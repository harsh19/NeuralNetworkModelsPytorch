import solver
import sys

mode = sys.argv[1]

import argparse
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', dest='mode')
    parser.add_argument("-hidden_size", type=int, dest="hidden_size", help="hidden_size", default=256)
    parser.add_argument("-num_of_points", type=int, dest="num_of_points", help="num_of_points", default=-1)
    parser.add_argument("-embeddings_size", type=int, dest="embeddings_size", help="embeddings_size", default=256)
    parser.add_argument("-epochs", type=int, dest="epochs", help="epochs", default=25)
    parser.add_argument("-batch_size", type=int, dest="batch_size", help="batch_size", default=25)
    parser.add_argument("-save_every_epoch", type=int, dest="save_every_epoch", help="save_every_epoch", default=1)
    parser.add_argument("-teacher_forcing_ratio", type=float, dest="teacher_forcing_ratio", help="teacher_forcing_ratio", default=0.0)    
    args = parser.parse_args()  
    return args

params = parseArguments()
trainer = solver.Solver(params)
trainer.main()
