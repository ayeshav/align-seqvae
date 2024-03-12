import numpy as np
from scipy.signal import decimate
from scipy.ndimage import gaussian_filter1d


def get_bins(event_time, start, end, binsize):

    bins = [np.arange(event_time[i] + start, event_time[i] + end, binsize) for i in range(len(event_time))]

    return np.stack(bins)


def get_spike_counts(spike_times, bins):
    return np.histogram(spike_times, bins)[0]


def get_masked_variable(var, bins):
    mask = (var[:, 0] >= bins[0]) & (var[:, 0] < bins[1])
    return var[mask, 1:]


def make_obs_mask(bins, obs_interval):
    mask = np.full(bins.shape, False)

    for i, (start, end) in enumerate(obs_interval):
        mask[i, (bins[i][0] < start) & (bins[i][1] > end)] = True

    return mask


def get_trialized_data(neurons, behavior, all_bins, all_bins_delay, trial_length, task='maze', binsize=0.02,
                       smooth_factor=2.5):
    spike_count_trial = []
    velocity_trial = []

    for i in range(len(all_bins)):
        spike_counts = [get_spike_counts(neurons[j]['ts'], all_bins_delay[i]) for j in range(len(neurons))]

        beh = get_masked_variable(behavior, (all_bins[i][0], all_bins[i][-1]))
        if task == 'maze':
            beh = np.diff(beh, axis=0)

        ds_velocity = decimate(beh, int(binsize * 1000), axis=0)[:trial_length]

        spike_count_trial.append(spike_counts)
        velocity_trial.append(ds_velocity)

    spike_count_trial = np.stack(spike_count_trial)
    velocity_trial = np.stack(velocity_trial)
    rates_trial = gaussian_filter1d(spike_count_trial.astype("float64"), smooth_factor, axis=1)

    return spike_count_trial, rates_trial, velocity_trial
