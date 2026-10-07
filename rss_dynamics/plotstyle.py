"""Common matplotlib style for the figures."""
import matplotlib.pyplot as plt

C = ['#2a6fdb', '#e0612a', '#2ba36b', '#a347c9', '#d1a21f', '#5a5a5a']
# nine legs: platform 1 legs in warm tones, platform 2 legs in cool tones
C9 = ['#b2182b', '#d6604d', '#f4a582', '#e08214', '#8c510a',
      '#2166ac', '#4393c3', '#35978f', '#01665e']


def style():
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False,
        'axes.spines.right': False, 'legend.frameon': False, 'figure.dpi': 100,
        'savefig.bbox': 'tight', 'mathtext.fontset': 'cm',
    })
