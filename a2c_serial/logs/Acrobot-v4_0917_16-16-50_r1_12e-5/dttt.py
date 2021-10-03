import matplotlib.pyplot as plt
import pandas as pd

d = {}
with open('episode-reward-loss-freq-sigma.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        episode, reward, loss, freq, sigma = map(float, line.split())
        d[int(episode)] = {'reward':reward, 'loss':loss, 'freq':freq, 'sigma':sigma}