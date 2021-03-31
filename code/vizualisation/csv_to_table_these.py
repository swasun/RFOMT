import copy

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os
import pandas as pd
import numpy as np
from pprint import pprint
import plotly.graph_objects as go
import plotly.io as pio
from collections import defaultdict

lst_skip_strategy = ["None", "OMP Distillation", "OMP Distillation w/o weights"]
lst_skip_task = ["correlation", "coherence"]
# lst_skip_task = []
lst_skip_subset = ["train/dev"]
# lst_skip_subset = []

tasks = [
    # "train_score",
    # "dev_score",
    "test_score",
    # "coherence",
    # "correlation"
]

dct_score_metric_fancy = {
    "accuracy_score": "% Accuracy",
    "mean_squared_error": "MSE"
}
dct_score_metric_best_fct = {
    "accuracy_score": np.argmax,
    "mean_squared_error": np.argmin
}

# dct_data_short = {
#     "Spambase": "Spambase",
#     "Diamonds": "Diamonds",
#     "Diabetes": "Diabetes",
#     "Steel Plates": "Steel P.",
#     "KR-VS-KP": "KR-VS-KP",
#     "Breast Cancer": "Breast C.",
#     "Kin8nm": "Kin8nm",
#     "LFW Pairs": "LFW P.",
#     "Gamma": "Gamma",
#     "California Housing": "California H.",
#     "Boston": "Boston",
# }

dct_data_short = {
    "Spambase": "Sp. B.",
    "Diamonds": "Diam.",
    "Diabetes": "Diab.",
    "Steel Plates": "St. P.",
    "KR-VS-KP": "KR-KP",
    "Breast Cancer": "B. C.",
    "Kin8nm": "Kin.",
    "LFW Pairs": "LFW P.",
    "Gamma": "Gam.",
    "California Housing": "C. H.",
    "Boston": "Bos.",
}

dct_data_best = {
    "Spambase": np.max,
    "Diamonds": np.min,
    "Diabetes": np.min,
    "Steel Plates": np.max,
    "KR-VS-KP": np.max,
    "Breast Cancer": np.max,
    "Kin8nm": np.min,
    "LFW Pairs": np.max,
    "Gamma": np.max,
    "California Housing": np.min,
    "Boston": np.min,
}
dct_data_metric = {
    "Spambase": "Acc.",
    "Diamonds": "MSE",
    "Diabetes": "MSE",
    "Steel Plates": "Acc.",
    "KR-VS-KP": "Acc.",
    "Breast Cancer": "Acc.",
    "Kin8nm": "MSE",
    "LFW Pairs": "Acc.",
    "Gamma": "Acc.",
    "California Housing": "MSE",
    "Boston": "MSE",
}



def get_max_from_df(df, best_fct):
    nb_to_consider = 10
    df.sort_values(by="forest_size", inplace=True)
    df_groupby_forest_size = df.groupby(['forest_size'])
    forest_sizes = list(df_groupby_forest_size["forest_size"].mean().values)[:nb_to_consider]
    mean_value = df_groupby_forest_size[task].mean().values[:nb_to_consider]
    std_value = df_groupby_forest_size[task].std().values[:nb_to_consider]

    try:
        argmax = best_fct(mean_value)
    except:
        print("no results", strat, data_name, task, subset_name)
        return -1, -1, -1

    max_mean = mean_value[argmax]
    max_std = std_value[argmax]
    max_forest_size = forest_sizes[argmax]

    return max_forest_size, max_mean, max_std



if __name__ == "__main__":

    load_dotenv(find_dotenv('.env'))
    dir_name = "bolsonaro_models_29-03-20_v3_2"
    dir_path = Path(os.environ["project_dir"]) / "results" / dir_name

    out_dir = Path(os.environ["project_dir"]) / "reports/figures" / dir_name

    input_dir_file = dir_path / "results.csv"
    df_results = pd.read_csv(open(input_dir_file, 'rb'))

    datasets = set(df_results["dataset"].values)
    strategies = sorted(list(set(df_results["strategy"].values) - set(lst_skip_strategy)))
    subsets = set(df_results["subset"].values)

    r"""
    \begin{table}[!h]
    \centering
    \begin{tabular}{l{}}
    \toprule
    \multicolumn{1}{c}{\textbf{Dataset}} & \textbf{Data dim.} $\datadim$        & \textbf{\# classes} & \textbf{Train size} $\nexamples$ & \textbf{Test size} $\nexamples'$ \\ \midrule
    \texttt{MNIST}~\cite{lecun-mnisthandwrittendigit-2010}                      & 784    & 10        & 60 000    & 10 000               \\ %\hline
    \texttt{Kddcup99}~\cite{Dua:2019}                                           & 116    & 23      & 4 893 431      & 5 000               \\ 
    \bottomrule
    \end{tabular}
    \caption{Main features of the datasets. Discrete, unordered attributes for dataset Kddcup99 have been encoded as one-hot attributes.}
    \label{table:data}
    \end{table}
    """


    for task in tasks:
        if task in lst_skip_task:
            continue

        dct_data_lst_tpl_results = defaultdict(lambda: [])

        lst_strats = []
        for data_name in datasets:
            df_data = df_results[df_results["dataset"] == data_name]
            score_metric_name = df_data["score_metric"].values[0]

            for subset_name in subsets:
                if subset_name in lst_skip_subset:
                    continue
                df_subset = df_data[df_data["subset"] == subset_name]

                ##################
                # all techniques #
                ##################
                for strat in strategies:
                    if strat in lst_skip_strategy:
                        continue
                    df_strat = df_subset[df_subset["strategy"] == strat]

                    if "OMP" in strat:
                        ###########################
                        # traitement without weights #
                        ###########################
                        df_strat_wo_weights = df_strat[df_strat["wo_weights"] == True]

                        strat_woweights = "{} w/o weights".format(strat)
                        dct_data_lst_tpl_results[data_name].append(get_max_from_df(df_strat_wo_weights, dct_score_metric_best_fct[score_metric_name]))
                        if strat_woweights not in lst_strats: lst_strats.append(strat_woweights)

                    #################################
                    # traitement general wo_weights #
                    #################################
                    df_strat_wo_weights = df_strat[df_strat["wo_weights"] == False]


                    dct_data_lst_tpl_results[data_name].append(get_max_from_df(df_strat_wo_weights, dct_score_metric_best_fct[score_metric_name]))
                    if strat not in lst_strats: lst_strats.append(strat)

                title = "{} {} {}".format(task, data_name, subset_name)

                # fig.show()
                sanitize = lambda x: x.replace(" ", "_").replace("/", "_").replace("+", "_")
                filename = sanitize(title)
                # output_dir = out_dir / sanitize(subset_name) / sanitize(task)
                # output_dir.mkdir(parents=True, exist_ok=True)
                # fig.write_image(str((output_dir / filename).absolute()) + ".png")


        # pprint(dct_data_lst_tpl_results)

        lst_data_ordered = [
            "Diamonds",
            "Diabetes",
            "Kin8nm",
            "California Housing",
            "Boston",
            "Spambase",
            "Steel Plates",
            "KR-VS-KP",
            "Breast Cancer",
            "LFW Pairs",
            "Gamma"
        ]


        arr_results_str = np.empty((len(lst_strats)+1, len(datasets) + 1 ), dtype="object")
        nb_spaces = 25
        dct_strat_str = defaultdict(lambda: [])
        s_empty = "{}" + " "*(nb_spaces-2) + " & "
        arr_results_str[0][0] = s_empty
        # arr_results_str[0][1] = s_empty
        for idx_data, data_name in enumerate(lst_data_ordered):
            lst_tpl_results = dct_data_lst_tpl_results[data_name]
            data_name_short = dct_data_short[data_name]
            # s_data_tmp = "{}".format(data_name_short)
            # add metric in parenthesis
            # s_data_tmp += "({})".format(dct_data_metric[data_name])
            # s_data_tmp = "\\texttt{{ {} }}".format(data_name_short)
            s_data_tmp = "\\multicolumn{{2}}{{c}}{{ \\texttt{{ {} }} }}".format(data_name)
            s_data_tmp += " "*(nb_spaces - len(s_data_tmp))
            s_data_tmp += " & "
            arr_results_str[0, idx_data + 1] = s_data_tmp


            array_results = np.array(lst_tpl_results)
            best_result_perf = dct_data_best[data_name](array_results[:, 1])
            best_result_perf_indexes = np.argwhere(array_results[:, 1] == best_result_perf)

            copye_array_results = copy.deepcopy(array_results)
            if dct_data_best[data_name] is np.min:
                copye_array_results[best_result_perf_indexes] = np.inf
            else:
                copye_array_results[best_result_perf_indexes] = -np.inf

            best_result_perf_2 = dct_data_best[data_name](copye_array_results[:, 1])
            best_result_perf_indexes_2 = np.argwhere(copye_array_results[:, 1] == best_result_perf_2)

            best_result_prune = np.min(array_results[:, 0])
            best_result_prune_indexes = np.argwhere(array_results[:, 0] == best_result_prune)

            for idx_strat, tpl_results in enumerate(array_results):
                str_strat = "\\texttt{{ {} }}".format(lst_strats[idx_strat])
                # str_strat = "\\multicolumn{{2}}{{c}}{{ \\texttt{{ {} }} }}".format(lst_strats[idx_strat])
                # str_strat = "\\multicolumn{{2}}{{c}}{{ \\thead{{ \\texttt{{ {} }} }} }}".format("}\\\\ \\texttt{".join(lst_strats[idx_strat].split(" ", 1)))
                # str_strat = "\\multicolumn{{2}}{{c}}{{ \\thead{{ {} }} }} ".format("\\\\".join(lst_strats[idx_strat].split(" ", 1)))
                str_strat += " " * (nb_spaces - len(str_strat)) + " & "
                arr_results_str[idx_strat+1, 0] =  str_strat

                # str_header = " {} & #tree &".format(dct_data_metric[data_name])
                # arr_results_str[idx_strat + 1, 1] = str_header

                best_forest_size = tpl_results[0]
                best_mean = tpl_results[1]
                best_std = tpl_results[2]
                if dct_data_metric[data_name] == "Acc.":
                    str_perf = "{:.2f}\\%".format(best_mean * 100)
                else:
                    str_perf = "{:.3E}".format(best_mean)

                str_prune = "{:d}".format(int(best_forest_size))

                if idx_strat in best_result_perf_indexes:
                    # str_formating = "\\textbf{{ {} }}".format(str_result_loc)
                    str_formating = "\\textbf[{}]"
                    # str_formating = "\\textbf{{ {:.3E} }}(\\~{:.3E})".format(best_mean, best_std)
                elif idx_strat in best_result_perf_indexes_2:
                    str_formating = "\\underline[{}]"
                    # str_formating = "\\underline{{ {:.3E} }}(\\~{:.3E})".format(best_mean, best_std)
                else:
                    str_formating = "{}"
                    # str_formating = "{:.3E}(~{:.3E})".format(best_mean, best_std)

                if idx_strat in best_result_prune_indexes:
                    str_formating = str_formating.format("\\textit[{}]")
                    # str_prune = " & \\textit{{ {:d} }}".format(int(best_forest_size))
                # else:
                #     str_prune = " & {:d}".format(int(best_forest_size))
                str_result = str_formating.format(str_perf) + " & " + str_formating.format(str_prune)
                str_result += " "*(nb_spaces - len(str_result))
                str_result = str_result.replace("[", "{").replace("]", "}")

                arr_results_str[idx_strat+1, idx_data+1] = str_result + " & "
                dct_strat_str[lst_strats[idx_strat]].append(str_result)

        # arr_results_str = arr_results_str.T

        arr_results_str_classif = arr_results_str[:, 6:]
        arr_results_str_classif = np.hstack([arr_results_str[:, 0:1], arr_results_str_classif])
        arr_results_str_reg = arr_results_str[:, :6]

        for arr_results_str in [arr_results_str_classif, arr_results_str_reg]:
            print(r"\toprule")
            for idx_lin, lin in enumerate(arr_results_str):
                if idx_lin == 1:
                    print("\\midrule")
                # if idx_lin == 6:
                #     print("\\midrule")
                # if lst_data_ordered[idx_lin-1] == "Diamonds":
                #     print("%", end="")
                line_print = " ".join(list(lin))
                line_print = line_print.rstrip(" &") + "\\\\"
                print(line_print)
            print(r"\bottomrule")
        # s_data = s_data.rstrip(" &") + "\\\\"
        # print(s_data)
        # for strat, lst_str_results in dct_strat_str.items():
        #     str_strat = "\\texttt{{ {} }}".format(strat)
        #     str_strat += " "*(nb_spaces - len(str_strat))
        #     str_strat += " & " + " & ".join(lst_str_results)
        #     str_strat += "\\\\"
        #     print(str_strat)

                # exit()
