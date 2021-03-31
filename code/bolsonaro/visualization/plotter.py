import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Plotter(object):

    @staticmethod
    def weight_density(all_experiment_weights, file_path):
        """
        Function that creates the figure with the density of the weights
        :param all_experiment_weights: The weights for the different experiments
        :param file path: str, path where the figure will be saved
        TODO: colored by seed number or not?
        TODO: represents both the seed AND the extracted tree information in the legend
        """

        """
        Convert dictionnary of structure
        {seed_1:[M x W]], seed_k:[M x W]}
        to numpy.ndarray with dim [K x M x W]
        where K is the seed number, M is the
        number of extracted trees and W the
        weight number.
        """
        all_experiment_weights = np.array(list(all_experiment_weights.values()))

        n = len(all_experiment_weights)

        """
        Get as many different colors from the specified cmap (here nipy_spectral)
        as there are seeds used.
        """
        colors = Plotter.get_colors_from_cmap(n)

        fig, ax = plt.subplots()
        # For each seed
        for i in range(n):
            # For each weight set of a given extracted tree number
            for weights in all_experiment_weights[i]:
                """
                Plot the series of weights that aren't zero,
                colored by seed number.
                """
                pd.Series(weights[np.nonzero(weights)]).plot.kde(
                    figsize=(15, 10), ax=ax, color=colors[i])

        ax.set_title('Density weights of the OMP')
        fig.savefig(file_path, dpi=fig.dpi)
        plt.close(fig)

    @staticmethod
    def plot_mean_and_CI(ax, mean, lb, ub, x_value, color_mean=None, facecolor=None, label=None):
        #print(x_value, mean, lb, ub)
        # plot the shaded range of the confidence intervals
        ax.fill_between(x_value, ub, lb, facecolor=facecolor, alpha=.5)
        # plot the mean on top
        ax.plot(x_value, mean, c=color_mean, label=label)

    @staticmethod
    def plot_stage1_losses(file_path, all_experiment_scores_with_params,
        all_experiment_scores_wo_params, x_value, xlabel, ylabel, all_labels, title):

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

        n = len(all_experiment_scores_with_params)

        if n != len(all_experiment_scores_wo_params):
            raise ValueError('all_experiment_scores_with_params and all_experiment_scores_wo_params must have the same len to be compared.')

        """
        Get as many different colors from the specified cmap (here nipy_spectral)
        as there are curve to plot.
        """
        colors = Plotter.get_colors_from_cmap(n)

        for j, all_experiment_scores in enumerate([all_experiment_scores_with_params,
            all_experiment_scores_wo_params]):
            # For each curve to plot
            for i in range(n):
                # Retreive the scores in a list for each seed
                experiment_scores = list(all_experiment_scores[i].values())
                # Compute the mean and the std for the CI
                mean_experiment_scores = np.average(experiment_scores, axis=0)
                std_experiment_scores = np.std(experiment_scores, axis=0)
                # Plot the score curve with the CI
                Plotter.plot_mean_and_CI(
                    ax=axes[j],
                    mean=mean_experiment_scores,
                    lb=mean_experiment_scores + std_experiment_scores,
                    ub=mean_experiment_scores - std_experiment_scores,
                    x_value=x_value,
                    color_mean=colors[i],
                    facecolor=colors[i],
                    label=all_labels[i]
                )

        axes[0].set_xlabel(xlabel)
        axes[1].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[1].set_title(title)
        handles, labels = axes[0].get_legend_handles_labels()
        legend = axes[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1, -0.15))
        fig.savefig(file_path, dpi=fig.dpi, bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_stage2_losses(file_path, all_experiment_scores, x_value,
        xlabel, ylabel, all_labels, title, filter_num=-1):

        fig, ax = plt.subplots()

        nb_experiments = len(all_experiment_scores)

        """
        Get as many different colors from the specified cmap (here nipy_spectral)
        as there are curve to plot.
        """
        colors = Plotter.get_colors_from_cmap(nb_experiments)

        # For each curve to plot
        for i in range(nb_experiments):
            # Retreive the scores in a list for each seed
            experiment_scores = list(all_experiment_scores[i].values())
            # Compute the mean and the std for the CI
            mean_experiment_scores = np.average(experiment_scores, axis=0)
            std_experiment_scores = np.std(experiment_scores, axis=0)

            # Plot the score curve with the CI
            Plotter.plot_mean_and_CI(
                ax=ax,
                mean=mean_experiment_scores,
                lb=mean_experiment_scores + std_experiment_scores,
                ub=mean_experiment_scores - std_experiment_scores,
                x_value=x_value[:filter_num] if len(mean_experiment_scores) == filter_num else x_value,
                color_mean=colors[i],
                facecolor=colors[i],
                label=all_labels[i]
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='upper right')
        fig.savefig(file_path, dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def get_colors_from_cmap(n_colors, colormap_name='nipy_spectral'):
        return [plt.get_cmap(colormap_name)(1. * i/n_colors) for i in range(n_colors)]
