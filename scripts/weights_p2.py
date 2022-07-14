import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_info(title, x_label, y_label, ticks=True):
    if ticks:
        ax = plt.gca()
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.draw()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='lower left')
    plt.show()


diff_steps = 1000
betas = np.linspace(0.0001, 0.02, diff_steps)
alphas = 1 - betas
alphas_bar = np.array([np.prod(alphas[:i]) for i in range(1, len(alphas) + 1)])
one_minus_alphas_bar = 1 - alphas_bar
snr = alphas_bar / one_minus_alphas_bar
snr_log10_space = np.log10(snr)[::-1]

vlb = np.array([1] * diff_steps)
vlb_norm = vlb / np.sum(vlb)

lambdas = np.array([2 * alphas[i] * one_minus_alphas_bar[i] / betas[i] for i in range(diff_steps)])

gamma_char = '\u03B3'
gammas = [0.0, 0.5, 1.0]

lambda_gammas = [lambdas * (1 - alphas_bar) ** gamma for gamma in gammas]

for gamma, lambda_gamma in enumerate(lambda_gammas):
    plt.plot(lambda_gamma, label=f'{gamma_char} = {gammas[gamma]}')
plot_info(title='Non normalized linear schedule',
          x_label='Diffusion Steps',
          y_label='Weights')

lambda_gammas_norm = [lambda_gamma / np.sum(lambda_gamma) for lambda_gamma in lambda_gammas]

for gamma, lambda_gamma_norm in enumerate(lambda_gammas_norm):
    plt.plot(lambda_gamma_norm, label=f'{gamma_char} = {gammas[gamma]}')
plt.plot(vlb_norm, linestyle='dashed', label='vlb')
plot_info(title='Normalized linear schedule',
          x_label='Diffusion Steps',
          y_label='Weights')

for gamma, lambda_gamma in enumerate(lambda_gammas):
    plt.plot(snr_log10_space[::-1], lambda_gamma, label=f'{gamma_char} = {gammas[gamma]}')
plot_info(title='Non normalized linear schedule',
          x_label='SNR (log10)',
          y_label='Weights',
          ticks=False)

for gamma, lambda_gamma_norm in enumerate(lambda_gammas_norm):
    plt.plot(snr_log10_space[::-1], lambda_gamma_norm, label=f'{gamma_char} = {gammas[gamma]}')
plt.plot(snr_log10_space[::-1], vlb_norm, linestyle='dashed', label='vlb')
plot_info(title='Normalized linear schedule',
          x_label='SNR (log10)',
          y_label='Weights',
          ticks=False)
