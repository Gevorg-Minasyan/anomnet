import argparse
from anomnet import AnomNet
import numpy as np
import pickle as pkl 


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='False')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--num_filters', type=int, default=16)
    parser.add_argument('--num_output_filter', type=int, default=12)
    parser.add_argument('--use_skip', type=int, default=1)
    parser.add_argument('--look_ahead', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_rate', type=float, default=0.0002)
    args = parser.parse_args(*argument_array)
    return args

if __name__ == "__main__":

    args = parse_args()


    model = AnomNet(input_size = args.input_size,
                in_channels = args.in_channels,
                num_filters = args.num_filters,
                num_output_filter = args.num_output_filter,
                use_skip = args.use_skip,
                look_ahead = args.look_ahead,
                epochs = args.epochs,
                lr_rate = args.lr_rate)   

    if args.train == 'True':

        # load DATA
        ts_df_list = pkl.load(open('./data/ts_list.pkl', 'rb'))
        ts_values = ts_df_list[0]['value'].values

        #train
        model.fit(ts_values)

        output = model.predict(None)
        pkl.dump(output, open('output.pkl', 'wb'))

    else:
        pass
           
