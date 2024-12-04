import numpy as np


def first_steps(g_scale, steps):
    g_scales = np.ones(50)
    g_scales[:steps] *= g_scale
    g_scales[steps:] = 0.
    return g_scales.tolist()


def last_steps(g_scale, steps):
    g_scales = np.ones(50)
    g_scales[-steps:] *= g_scale
    g_scales[:-steps] = 0.
    return g_scales.tolist()

def first_steps_linear(g_scale, steps):
    g_scales = np.linspace(0, g_scale, num=steps).tolist()
    g_scales += [g_scale] * (50 - steps)
    return g_scales

def last_steps_linear(g_scale, steps):
    g_scales = [g_scale] * (50 - steps)
    g_scales += np.linspace(g_scale, 0, num=steps).tolist()
    return g_scales
