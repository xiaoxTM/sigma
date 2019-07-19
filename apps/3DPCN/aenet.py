#! /usr/bin/env python3

from . import auto_encoder as ate

def point_capsule_networks(inputs):
    x = ate.encode(inputs)
    latent_capsules = ops._agreement_routing(x)
    x = ate.decode(latent_capsules)

    return latent_capsules, x


def train():
    pass


def predict():
    pass
