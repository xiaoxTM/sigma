import numpy as np

def make_wheel(fitness):
    wheel = []
    probs = fitness / fitness.sum()
    top = 0
    for i,p in enumerate(probs):
        wheel.append((top, top+p, i))
        top += p
    return wheel

def bin_search(wheel, prob):
    mid = len(wheel) // 2
    low, high, answer = wheel[mid]
    if low <= prob <= high:
        return answer
    elif high < prob:
        return bin_search(wheel[mid+1:],prob)
    else:
        return bin_search(wheel[:mid], prob)

def stochastic_universal_sample(wheel, num_samples, sample_times):
    if sample_times is None:
        sample_times = len(wheel) // 2
    step_size = 1.0 / num_samples
    samples = np.zeros((sample_times, num_samples), dtype=np.int32)
    for time in range(sample_times):
        r = np.random.random()
        samples[time][0] = bin_search(wheel,r)
        for i in range(1, num_samples):
            r += step_size
            if r > 1:
                r %= 1
            samples[time][i] = bin_search(wheel, r)
    return samples

def sus(fitness, num_samples, sample_times):
    wheel = make_wheel(fitness)
    samples = stochastic_universal_sample(wheel,num_samples,sample_times)
    return samples
