from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.special import softmax
from sklearn import svm
from sklearn.linear_model import LinearRegression

lst_skip_strategy = ["None", "OMP Distillation", "OMP Distillation w/o weights"]
# lst_skip_subset = ["train/dev"]
lst_task_train_dev = ["coherence", "correlation"]

tasks = [
    # "train_score",
    "dev_score",
    "test_score",
    # "coherence",
    # "correlation",
    # "negative-percentage",
    # "dev_strength",
    # "test_strength",
    # "dev_correlation",
    # "test_correlation",
    # "dev_coherence",
    # "test_coherence",
    # "negative-percentage-test-score"
]

dct_score_metric_fancy = {
    "accuracy_score": "% de Précision",
    "mean_squared_error": "MSE"
}

pio.templates.default = "plotly_white"

dct_color_by_strategy = {
    "OMP": (255, 117, 26), # orange
    "NN-OMP": (255, 0, 0), # red
    "OMP Distillation": (255, 0, 0), # red
    "OMP Distillation w/o weights": (255, 0, 0), # red
    "OMP w/o weights": (255, 117, 26), # orange
    "NN-OMP w/o weights": (255, 0, 0), # grey
    "Random": (128, 128, 128), # black
    "Zhang Similarities": (255,105,180), # rose
    'Zhang Predictions': (128, 0, 128), # turquoise
    'Ensemble': (0, 0, 255), # blue
    "Kmeans": (0, 255, 0) # red
}

dct_data_color = {
    "Boston": (255, 117, 26),
    "Breast Cancer": (255, 0, 0),
    "California Housing": (255,105,180),
    "Diabetes": (128, 0, 128),
    "Diamonds": (0, 0, 255),
    "Kin8nm": (128, 128, 128),
    "KR-VS-KP": (0, 255, 0),
    "Spambase": (0, 128, 0),
    "Steel Plates": (128, 0, 0),
    "Gamma": (0, 0, 128),
    "LFW Pairs": (64, 64, 64),
}

dct_dash_by_strategy = {
    "OMP": "solid",
    "NN-OMP": "solid",
    "OMP Distillation": "dash",
    "OMP Distillation w/o weights": "dash",
    "OMP w/o weights": "dot",
    "NN-OMP w/o weights": "dot",
    "Random": "longdash",
    "Zhang Similarities": "dash",
    'Zhang Predictions': "dash",
    'Ensemble': "dash",
    "Kmeans": "dash"
}

dct_symbol_by_strategy = {
    "OMP": "x",
    "NN-OMP": "star",
    "OMP Distillation": "x",
    "OMP Distillation w/o weights": "x",
    "OMP w/o weights": "x",
    "NN-OMP w/o weights": "star",
    "Random": "x",
    "Zhang Similarities": "hexagon",
    'Zhang Predictions': "hexagon2",
    'Ensemble': "pentagon",
    "Kmeans": "octagon",
}

def get_index_of_first_last_repeted_elemen(iterabl):
    last_elem = iterabl[-1]
    reversed_idx = 0
    for idx, elm in enumerate(iterabl[::-1]):
        if elm != last_elem:
            break
        reversed_idx = -(idx+1)

    index_flat = len(iterabl) + reversed_idx
    return index_flat

GLOBAL_TRACE_TO_ADD_LAST = None

def add_trace_from_df(df, fig, task, strat, stop_on_flat=False):
    global GLOBAL_TRACE_TO_ADD_LAST

    df.sort_values(by="forest_size", inplace=True)
    df_groupby_forest_size = df.groupby(['pruning_percent'])
    forest_sizes = list(df_groupby_forest_size["pruning_percent"].mean().values)
    mean_value = df_groupby_forest_size[task].mean().values
    std_value = df_groupby_forest_size[task].std().values

    index_flat = len(forest_sizes)
    if stop_on_flat:
        actual_forest_sizes = list(df_groupby_forest_size["actual-forest-size"].mean().values)
        index_flat = get_index_of_first_last_repeted_elemen(actual_forest_sizes)
        # for this trace to appear on top of all others
        GLOBAL_TRACE_TO_ADD_LAST = go.Scatter(
                    mode='markers',
                    x=[forest_sizes[index_flat-1]],
                    y=[mean_value[index_flat-1]],
                    marker_symbol="star",
                    marker=dict(
                        color="rgb{}".format(dct_color_by_strategy[strat]),
                        size=15,
                        line=dict(
                            color='Black',
                            width=2
                        )
                    ),
                    name="Final NN-OMP",
                    showlegend=True
                )

    forest_sizes = forest_sizes[:index_flat]
    mean_value = mean_value[:index_flat]
    std_value = std_value[:index_flat]
    std_value_upper = list(mean_value + std_value)
    std_value_lower = list(mean_value - std_value)
    # print(df_strat)
    fig.add_trace(go.Scatter(x=forest_sizes, y=mean_value,
                             mode='lines',
                             name=strat,
                             line=dict(dash=dct_dash_by_strategy[strat], color="rgb{}".format(dct_color_by_strategy[strat]))
                             ))

    fig.add_trace(go.Scatter(
        x=forest_sizes + forest_sizes[::-1],
        y=std_value_upper + std_value_lower[::-1],
        fill='toself',
        showlegend=False,
        fillcolor='rgba{}'.format(dct_color_by_strategy[strat] + tpl_transparency),
        line_color='rgba(255,255,255,0)',
        name=strat
    ))

tpl_transparency = (0.1,)

dct_metric_lambda_prop_amelioration = {
    "accuracy_score": (lambda mean_value_acc, mean_value_random_acc: (mean_value_acc - mean_value_random_acc) / mean_value_random_acc),
    "mean_squared_error": (lambda mean_value_mse, mean_value_random_mse: (mean_value_random_mse - mean_value_mse) / mean_value_random_mse)
}

dct_metric_figure = {
    "accuracy_score":go.Figure(),
    "mean_squared_error": go.Figure()
}

dct_gamma_by_dataset = {
    "Boston": 5,
    "Breast Cancer": 5,
    "California Housing": 5,
    "Diabetes": 5,
    "Diamonds": 5,
    "Kin8nm": 5,
    "KR-VS-KP": 5,
    "Spambase": 5,
    "Steel Plates": 5,
    "Gamma": 5,
    "LFW Pairs": 5,
}

def base_figures(skip_NN=False):

    for task in tasks:
        for data_name in datasets:
            df_data = df_results[df_results["dataset"] == data_name]
            score_metric_name = df_data["score_metric"].values[0]

            # This figure is for basic representation: task metric wrt the number of pruned tree
            fig = go.Figure()

            ##################
            # all techniques #
            ##################
            for strat in strategies:
                if strat in lst_skip_strategy or (skip_NN and "NN-OMP" in strat):
                    continue

                # if task == "negative-percentage-test-score":
                #     if strat == "OMP":
                #         df_strat = df_data[df_data["strategy"] == strat]
                #         df_strat = df_strat[df_strat["subset"] == "train+dev/train+dev"]
                #         df_strat_wo_weights = df_strat[df_strat["wo_weights"] == False]
                #
                #         df_groupby_forest_size = df_strat_wo_weights.groupby(['forest_size'])
                #
                #
                #         forest_sizes = df_groupby_forest_size["forest_size"].mean().values
                #         x_values = df_groupby_forest_size["negative-percentage"].mean().values
                #         y_values = df_groupby_forest_size["test_score"].mean().values
                #         # print(df_strat)
                #         fig.add_trace(go.Scatter(x=x_values, y=y_values,
                #                                  mode='markers',
                #                                  name=strat,
                #                                  # color=forest_sizes,
                #                                  marker=dict(
                #                                     # size=16,
                #                                     # cmax=39,
                #                                     # cmin=0,
                #                                     color=forest_sizes,
                #                                     colorbar=dict(
                #                                         title="Forest Size"
                #                                     ),
                #                                     # colorscale="Viridis"
                #                                 ),
                #                                  # marker=dict(color="rgb{}".format(dct_color_by_strategy[strat]))
                #          ))
                #
                #     continue


                df_strat = df_data[df_data["strategy"] == strat]
                df_strat = df_strat[df_strat["subset"] == "train+dev/train+dev"]
                # df_strat = df_strat[df_strat["subset"] == "train/dev"]

                if "OMP" in strat:
                    ###########################
                    # traitement avec weights #
                    ###########################
                    df_strat_wo_weights = df_strat[df_strat["wo_weights"] == False]
                    if strat == "NN-OMP":
                        add_trace_from_df(df_strat_wo_weights, fig, task, strat, stop_on_flat=True)
                    else:
                        add_trace_from_df(df_strat_wo_weights, fig, task, strat)


                #################################
                # traitement general wo_weights #
                #################################
                if "OMP" in strat:
                    df_strat_wo_weights = df_strat[df_strat["wo_weights"] == True]
                else:
                    df_strat_wo_weights = df_strat[df_strat["wo_weights"] == False]

                if "OMP" in strat:
                    strat = "{} w/o weights".format(strat)

                if strat == "NN-OMP":
                    add_trace_from_df(df_strat_wo_weights, fig, task, strat,  stop_on_flat=True)
                else:
                    add_trace_from_df(df_strat_wo_weights, fig, task, strat)

            title = "{} {}".format(task, data_name)
            yaxis_title = "% negative weights" if task == "negative-percentage" else dct_score_metric_fancy[score_metric_name]
            xaxis_title = "% negative weights" if task == "negative-percentage-test-score" else "% d'Arbres sélectionnés"

            if not skip_nn:
                fig.add_trace(GLOBAL_TRACE_TO_ADD_LAST)
            fig.update_layout(barmode='group',
                              # title=title,
                              xaxis_title=xaxis_title,
                              yaxis_title=yaxis_title,
                              font=dict(
                                  # family="Courier New, monospace",
                                  size=24,
                                  color="black"
                              ),
                                showlegend = False,
                                margin = dict(
                                    l=1,
                                    r=1,
                                    b=1,
                                    t=1,
                                    # pad=4
                                ),
                              legend=dict(
                                  traceorder="normal",
                                  font=dict(
                                      family="sans-serif",
                                      size=24,
                                      color="black"
                                  ),
                                  # bgcolor="LightSteelBlue",
                                  # bordercolor="Black",
                                  borderwidth=1,
                              )
                              )
            # fig.show()
            if skip_NN:
                str_no_nn = " no nn"
                title += str_no_nn
            sanitize = lambda x: x.replace(" ", "_").replace("/", "_").replace("+", "_")
            filename = sanitize(title)
            output_dir = out_dir / sanitize(task)
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.update_xaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)
            fig.update_yaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)

            fig.write_image(str((output_dir / filename).absolute()) + ".png")

def global_figure():
    for task in tasks:

        for metric in ["accuracy_score", "mean_squared_error"]:

            # fig = go.Figure()
            df_data = df_results

            df_strat_random = df_data[df_data["strategy"] == "Random"]
            df_strat_random = df_strat_random[df_strat_random["subset"] == "train+dev/train+dev"]
            df_strat_random_wo_weights = df_strat_random[df_strat_random["wo_weights"] == False]
            df_strat_random_wo_weights.sort_values(by="pruning_percent", inplace=True)

            # df_strat_random_wo_weights_acc = df_strat_random_wo_weights[df_strat_random_wo_weights["score_metric"] == "accuracy_score"]
            # df_groupby_random_forest_size_acc = df_strat_random_wo_weights_acc.groupby(['pruning_percent'])
            # forest_sizes_random_acc = df_groupby_random_forest_size_acc["pruning_percent"].mean().values
            # mean_value_random_acc = df_groupby_random_forest_size_acc[task].mean().values

            df_strat_random_wo_weights_mse = df_strat_random_wo_weights[df_strat_random_wo_weights["score_metric"] == metric]
            # df_strat_random_wo_weights_mse = df_strat_random_wo_weights[df_strat_random_wo_weights["score_metric"] == "mean_squared_error"]
            df_groupby_random_forest_size_mse = df_strat_random_wo_weights_mse.groupby(['pruning_percent'])
            forest_sizes_random_mse = df_groupby_random_forest_size_mse["pruning_percent"].mean().values
            # assert np.allclose(forest_sizes_random_acc, forest_sizes_random_mse)
            mean_value_random_mse = df_groupby_random_forest_size_mse[task].mean().values


            for strat in strategies:
                if strat in lst_skip_strategy or strat == "Random":
                    continue

                df_strat = df_data[df_data["strategy"] == strat]
                df_strat = df_strat[df_strat["subset"] == "train+dev/train+dev"]
                df_strat_wo_weights = df_strat[df_strat["wo_weights"] == False]
                df_strat_wo_weights.sort_values(by="pruning_percent", inplace=True)

                # "accuracy_score"
                # "mean_squared_error"

                # df_accuracy = df_strat_wo_weights[df_strat_wo_weights["score_metric"] == "accuracy_score"]
                # df_groupby_forest_size = df_accuracy.groupby(['pruning_percent'])
                # forest_sizes_acc = df_groupby_forest_size["pruning_percent"].mean().values
                # mean_value_acc = df_groupby_forest_size[task].mean().values
                # propo_ameliration_mean_value_acc = (mean_value_acc - mean_value_random_acc)/mean_value_random_acc

                df_mse = df_strat_wo_weights[df_strat_wo_weights["score_metric"] == metric]
                # df_mse = df_strat_wo_weights[df_strat_wo_weights["score_metric"] == "mean_squared_error"]
                df_groupby_forest_size_mse = df_mse.groupby(['pruning_percent'])
                forest_sizes_mse = df_groupby_forest_size_mse["pruning_percent"].mean().values
                # assert np.allclose(forest_sizes_mse, forest_sizes_acc)
                # assert np.allclose(forest_sizes_random_acc, forest_sizes_acc)
                mean_value_mse = df_groupby_forest_size_mse[task].mean().values
                # propo_ameliration_mean_value_mse = (mean_value_random_mse - mean_value_mse) / mean_value_random_mse
                propo_ameliration_mean_value_mse = dct_metric_lambda_prop_amelioration[metric](mean_value_mse, mean_value_random_mse)

                # mean_value = np.mean([propo_ameliration_mean_value_acc, propo_ameliration_mean_value_mse], axis=0)
                mean_value = np.mean([propo_ameliration_mean_value_mse], axis=0)

                # std_value = df_groupby_forest_size[task].std().values
                # print(df_strat)
                dct_metric_figure[metric].add_trace(go.Scatter(x=forest_sizes_mse, y=mean_value,
                                         mode='markers',
                                         name=strat,
                                         # marker=dict(color="rgb{}".format(dct_color_by_strategy[strat])),
                                         marker_symbol = dct_symbol_by_strategy[strat],
                                        marker = dict(
                                            color="rgb{}".format(dct_color_by_strategy[strat]),
                                            size=20,
                                            # line=dict(
                                            #     color='Black',
                                            #     width=2
                                            # )
                                        ),
                                         ))

            title_global_figure = "Global {} {}".format(task, metric)
            sanitize = lambda x: x.replace(" ", "_").replace("/", "_").replace("+", "_")
            filename = sanitize(title_global_figure)


            dct_metric_figure[metric].update_layout(title=filename)
            dct_metric_figure[metric].write_image(str((out_dir / filename).absolute()) + ".png")
            # fig.show()

def weights_wrt_size():
    # lst_skip_data_weight_effect = ["Gamma", "KR-VS-KP", "Steel Plates"]
    lst_skip_data_weight_effect = ["Gamma"]
    fig = go.Figure()

    for data_name in datasets:

        if data_name in lst_skip_data_weight_effect:
            continue
        df_data = df_results[df_results["dataset"] == data_name]
        score_metric_name = df_data["score_metric"].values[0]

        ##################
        # all techniques #
        ##################
        strat = "OMP"
        df_strat = df_data[df_data["strategy"] == strat]
        df_strat = df_strat[df_strat["subset"] == "train+dev/train+dev"]
        df_strat_wo_weights = df_strat[df_strat["wo_weights"] == False]

        df_strat_wo_weights.sort_values(by="pruning_percent", inplace=True)

        df_groupby_forest_size = df_strat_wo_weights.groupby(['forest_size'])

        y_values = df_groupby_forest_size["negative-percentage"].mean().values
        y_values = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))

        x_values = df_groupby_forest_size["pruning_percent"].mean().values
        # x_values = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))

        # if score_metric_name == "mean_squared_error":
        #     y_values = 1/y_values

        lin_reg = svm.SVR(gamma=10)
        lin_reg.fit(x_values[:, np.newaxis], y_values)

        # xx = np.linspace(0, 1)
        yy = lin_reg.predict(x_values[:, np.newaxis])

        # print(df_strat)
        fig.add_trace(go.Scatter(x=x_values, y=y_values,
                                 mode='markers',
                                 name=strat,
                                 # color=forest_sizes,
                                 marker=dict(
                                     # size=16,
                                     # cmax=39,
                                     # cmin=0,
                                     color="rgb{}".format(dct_data_color[data_name]),
                                     # colorbar=dict(
                                     #     title="Forest Size"
                                     # ),
                                     # colorscale="Viridis"
                                 ),
                                 # marker=dict(color="rgb{}".format(dct_color_by_strategy[strat]))
                                 ))
        fig.add_trace(go.Scatter(x=x_values, y=yy,
                                 mode='lines',
                                 name=strat,
                                 # color=forest_sizes,
                                 marker=dict(
                                     # size=16,
                                     # cmax=39,
                                     # cmin=0,
                                     color="rgba{}".format(tuple(list(dct_data_color[data_name]) + [0.5])),
                                     # colorbar=dict(
                                     #     title="Forest Size"
                                     # ),
                                     # colorscale="Viridis"
                                 ),
                                 # marker=dict(color="rgb{}".format(dct_color_by_strategy[strat]))
                                 ))



    title = "{}".format("weight wrt size")

    fig.update_layout(barmode='group',
                      # title=title,
                      xaxis_title="% d'Arbres selectionnés",
                      yaxis_title="% de poids négatifs standardisé",
                      font=dict(
                          # family="Courier New, monospace",
                          size=24,
                          color="black"
                      ),
                      showlegend = False,
                      margin=dict(
                          l=1,
                          r=1,
                          b=3,
                          t=10,
                          # pad=4
                      ),
                      legend=dict(
                          traceorder="normal",
                          font=dict(
                              family="sans-serif",
                              size=24,
                              color="black"
                          ),
                          # bgcolor="LightSteelBlue",
                          # bordercolor="Black",
                          borderwidth=1,
                      )
                      )
    # fig.show()
    sanitize = lambda x: x.replace(" ", "_").replace("/", "_").replace("+", "_")
    filename = sanitize(title)
    output_dir = out_dir / sanitize(title)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.update_xaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)

    fig.write_image(str((output_dir / filename).absolute()) + ".png")

def effect_of_weights_figure():
    lst_skip_data_weight_effect = ["Gamma"]
    # lst_skip_data_weight_effect = ["Gamma", "KR-VS-KP", "Steel Plates"]

    fig = go.Figure()

    for data_name in datasets:
        #
        # if data_name in lst_skip_data_weight_effect:
        #     continue
        df_data = df_results[df_results["dataset"] == data_name]
        score_metric_name = df_data["score_metric"].values[0]

        ##################
        # all techniques #
        ##################
        strat = "OMP"
        df_strat = df_data[df_data["strategy"] == strat]
        df_strat = df_strat[df_strat["subset"] == "train+dev/train+dev"]
        df_strat_wo_weights = df_strat[df_strat["wo_weights"] == False]
        df_strat_wo_weights.sort_values(by="pruning_percent", inplace=True)

        df_groupby_forest_size = df_strat_wo_weights.groupby(['forest_size'])

        x_values = df_groupby_forest_size["negative-percentage"].mean().values
        y_values = df_groupby_forest_size["test_score"].mean().values
        if score_metric_name == "mean_squared_error":
            y_values = 1/y_values


        x_values = x_values[3:]
        y_values = y_values[3:]

        x_values = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
        y_values = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))

        # bins = np.histogram(x_values)[1]
        # indices_x_values = np.digitize(x_values, bins)-1
        # mean_val = np.empty(len(bins)-1)
        # for idx_group in range(len(bins) - 1):
        #     mean_val[idx_group] = np.mean(y_values[indices_x_values == idx_group])

        # lin_reg = LinearRegression()
        # lin_reg = svm.SVR(gamma=dct_gamma_by_dataset[data_name])
        lin_reg = svm.SVR(gamma=1.)
        lin_reg.fit(x_values[:, np.newaxis], y_values)

        xx = np.linspace(0, 1)
        yy = lin_reg.predict(xx[:, np.newaxis])



        # print(df_strat)
        fig.add_trace(go.Scatter(x=x_values, y=y_values,
                                 mode='markers',
                                 name=strat,
                                 showlegend=False,
                                 # color=forest_sizes,
                                 marker=dict(
                                     # size=16,
                                     # cmax=39,
                                     # cmin=0,
                                     color="rgb{}".format(dct_data_color[data_name]),
                                     # colorbar=dict(
                                     #     title="Forest Size"
                                     # ),
                                     # colorscale="Viridis"
                                 ),
                                 # marker=dict(color="rgb{}".format(dct_color_by_strategy[strat]))
                                 ))
        fig.add_trace(go.Scatter(x=xx, y=yy,
                                 mode='lines',
                                 name=data_name,
                                 # color=forest_sizes,
                                 marker=dict(
                                     # size=16,
                                     # cmax=39,
                                     # cmin=0,
                                     color="rgba{}".format(tuple(list(dct_data_color[data_name]) + [0.5])),
                                     # colorbar=dict(
                                     #     title="Forest Size"
                                     # ),
                                     # colorscale="Viridis"
                                 ),
                                 # marker=dict(color="rgb{}".format(dct_color_by_strategy[strat]))
                                 ))




    title = "{}".format("negative weights effect")

    fig.update_layout(barmode='group',
                      # title=title,
                      xaxis_title="% de poids négatifs standardisé",
                      yaxis_title="Performance standardisée",
                      font=dict(
                          # family="Courier New, monospace",
                          size=24,
                          color="black"
                      ),
                      showlegend = False,
                      margin=dict(
                          l=1,
                          r=1,
                          b=1,
                          t=1,
                          # pad=4
                      ),
                      legend=dict(
                          traceorder="normal",
                          font=dict(
                              family="sans-serif",
                              size=24,
                              color="black"
                          ),
                          # bgcolor="LightSteelBlue",
                          # bordercolor="Black",
                          borderwidth=1,
                      )
                      )
    # fig.show()
    sanitize = lambda x: x.replace(" ", "_").replace("/", "_").replace("+", "_")
    filename = sanitize(title)
    output_dir = out_dir / sanitize(title)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.update_xaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)

    fig.write_image(str((output_dir / filename).absolute()) + ".png")

if __name__ == "__main__":

    load_dotenv(find_dotenv('.env'))
    dir_name = "bolsonaro_models_29-03-20_v3_2"
    dir_path = Path(os.environ["project_dir"]) / "results" / dir_name

    out_dir = Path(os.environ["project_dir"]) / "reports/figures" / dir_name

    input_dir_file = dir_path / "results.csv"
    df_results = pd.read_csv(open(input_dir_file, 'rb'))

    datasets = set(df_results["dataset"].values)
    strategies = set(df_results["strategy"].values)
    subsets = set(df_results["subset"].values)

    for skip_nn in [True, False]:
        base_figures(skip_nn)
    effect_of_weights_figure()
    weights_wrt_size()
    # global_figure()
