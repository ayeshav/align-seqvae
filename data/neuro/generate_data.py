import os
import pickle
import torch
import re
from utils import *


def process_co_reaching_data():
    datapath = 'pickled'
    savepath = 'processed/'

    "Load default config for data"
    align_event = 'move_begins_time'

    start = - 0.140
    end = 0.561
    delay = 0.100
    binsize = 0.02

    smooth_ms = 0.05

    smooth_factor = smooth_ms / binsize
    trial_length = int((end - start) // binsize)

    data_co = {}

    for file in os.listdir(datapath):
        if file.endswith('.pkl'):
            with open(os.path.join(datapath, file), 'rb') as f:
                data = pickle.load(f)
        else:
            continue

        table = data['trial_table']
        neurons = data['out_struct']['units']
        vel = np.array(data['out_struct']['vel'])

        event_indices = {'target_presentation_time': 7,
                         'go_cue_time': 8,
                         'move_begins_time': 9}

        all_bins = get_bins(table[:, event_indices[align_event]], start, end, binsize)
        all_bins_delay = get_bins(table[:, event_indices[align_event]], start - delay, end - delay, binsize)

        spikes, rates, velocity = get_trialized_data(neurons, vel, all_bins, all_bins_delay, trial_length,
                                                     task='centre_out', binsize=binsize, smooth_factor=smooth_factor)

        session_name = re.search(r'full-(.*?).pkl', file).group(1)

        data_co[session_name] = {'y': torch.from_numpy(spikes).float(),
                                 'rates': torch.from_numpy(rates).float(),
                                 'velocity': torch.from_numpy(velocity).float(),
                                 'target': torch.from_numpy(table[:, 1].astype(int)),
                                 'angle': torch.from_numpy(table[:, 2]).float()}

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    torch.save(data_co, savepath + f'co_reaching_{align_event}.pt')


if __name__ == "__main__":

    process_co_reaching_data()
