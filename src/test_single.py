import torch
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-checkpoint", type=str)

args = arg_parser.parse_args()

checkpoint = torch.load(open(args.checkpoint, "rb"))
# keys = ['opt', 'optim', 'vocab', 'epoch', 'moving_avg', 'model']

model = checkpoint['model']

