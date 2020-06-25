#!/usr/bin/env python

import json
import sys

from matplotlib import pyplot as plt
import numpy as np

from asreview import ASReviewData

from asreviewcontrib.simulation.error import ErrorEntryPoint


def compute_rand_prob(n_sample, n_total, n_inclusions):
    cur_prob = np.zeros(n_inclusions+1)
    cur_prob[0] = 1
    for i_sample in range(n_sample):
        carry = 0
        for i_inc in range(n_inclusions+1):
            if n_total-i_sample < n_inclusions or n_total-i_sample == 0:
                continue
            inc_rate = (n_inclusions-i_inc)/(n_total-i_sample)
            new_carry = cur_prob[i_inc] * inc_rate
            cur_prob[i_inc] = carry + (1-inc_rate)*cur_prob[i_inc]
            carry = new_carry

    avg_bias = np.sum(cur_prob*np.arange(n_inclusions+1)*(n_total/n_sample)) - n_inclusions
    avg_dev = np.sum(cur_prob*np.abs(np.arange(n_inclusions+1)*(n_total/n_sample) - n_inclusions))
    return avg_bias, avg_dev


def get_avg_err(all_errors):
    inclusion_err = []
    inclusion_avg = []
    for query_i in range(1, len(all_errors[0]["inclusion_est"])):
        inclusion_est = [all_errors[i]["inclusion_est"][query_i]
                         for i in range(10)]
        inclusion_err.append(np.sqrt(np.var(inclusion_est)))
        inclusion_avg.append(np.mean(inclusion_est))

    perc_rev = all_errors[0]["perc_reviewed"][1:]
    return perc_rev, inclusion_avg, inclusion_err


def get_avg_err_prob(all_errors):
    prob_err = []
    prob_avg = []
    for query_i in range(1, len(all_errors[0]["prob_finished"])):
        prob_est = [all_errors[i]["prob_finished"][query_i]
                    for i in range(10)]
        prob_err.append(np.sqrt(np.var(prob_est)))
        prob_avg.append(np.mean(prob_est))

    perc_rev = all_errors[0]["perc_reviewed"][1:]
    return perc_rev, prob_avg, prob_err


def plot_avg_bias(all_errors, n_total_papers, dataset, n_sample=50):
    bias_new = []
    bias_random = []
    dev_new = []
    dev_random = []
    n_inclusions = all_errors[0]["n_total_inclusions"]
    perc_rev = all_errors[0]["perc_reviewed"][1:]
    for query_i in range(1, len(all_errors[0]["inclusion_est"])):
        inclusion_est = np.array([all_errors[i]["inclusion_est"][query_i]
                                  for i in range(10)])
        bias_new.append(np.mean(inclusion_est)
                        - all_errors[0]["n_total_inclusions"])
        dev_new.append(np.mean(np.abs(inclusion_est-all_errors[0]["n_total_inclusions"])))
        biases = []
        devs = []
        for error_data in all_errors:
            cur_inc = error_data["cur_included"][query_i]
            bias, dev = compute_rand_prob(
                n_sample,
                round((1-error_data["perc_reviewed"][query_i]/100)*n_total_papers),
                n_inclusions-cur_inc)
            biases.append(bias)
            devs.append(dev)
        bias_random.append(np.mean(biases))
        dev_random.append(np.mean(devs))
#             return

    plt.grid()
    plt.plot(perc_rev, bias_new, color="red", label="new - bias")
    plt.plot(perc_rev, bias_random, color="blue", label="random - bias")
    plt.plot(perc_rev, dev_new, color="red", linestyle=":", label="new - deviation")
    plt.plot(perc_rev, dev_random, color="blue", linestyle=":", label="random - deviation")
    plt.legend()
    plt.xlabel("% papers reviewed")
    plt.ylabel("# of papers")
    plt.title(f"{dataset} - {n_inclusions} inclusions - {n_sample} samples")
    plt.show()


def plot_avg_err(all_errors, dataset):
    n_inclusions = all_errors[0]["n_total_inclusions"]
    perc_rev, inclusion_avg, inclusion_err = get_avg_err(all_errors)

    plt.grid()
    plt.errorbar(perc_rev, inclusion_avg, inclusion_err, label="estimated")
    plt.plot(perc_rev, all_errors[0]["cur_included"][1:], color="red", label="found")
    plt.xlabel("% of papers reviewed")
    plt.ylabel("# of inclusions")
    plt.title(f"{dataset} - {n_inclusions} inclusions")
    plt.legend()
    plt.show()


def plot_avg_err_prob(all_errors, dataset):
    n_inclusions = all_errors[0]["n_total_inclusions"]

    perc_rev, prob_avg, prob_err = get_avg_err_prob(all_errors)
    plt.errorbar(perc_rev, prob_avg, prob_err)
    avg_included = []
    for query_i in range(1, len(all_errors[0]["cur_included"])):
        avg_included.append(
            np.mean([
                all_errors[i]["cur_included"][query_i]
                for i in range(len(all_errors))
            ])/all_errors[0]["n_total_inclusions"]
        )

    plt.grid()
    plt.errorbar(perc_rev, prob_avg, prob_err, label="estimated finished probability")
    plt.plot(perc_rev, avg_included, label="fraction inclusions found")
    plt.xlabel("% of papers reviewed")
    plt.title(f"{dataset}")
    plt.legend()
    plt.show()


def simulate_error(dataset="ace", plot=True):
    entry = ErrorEntryPoint()

    state_fp = f"output/simulation/{dataset}/results_0.h5"
    data_fp = f"data/{dataset}.csv"
    with open("output/optimization.json", "r") as f:
        opt_results = json.load(f)

    as_data = ASReviewData.from_file(data_fp)
    n_total_papers = len(as_data)
    all_errors = []

    for i_try_error in range(10):
        error_fp = f"output/error/{dataset}/error_0_{i_try_error}.json"
        try:
            with open(error_fp, "r") as f:
                error_data = json.load(f)
        except FileNotFoundError:
            error_data = entry.error_estimate(state_fp, data_fp, opt_results)
            with open(error_fp, "w") as f:
                json.dump(error_data, f)

        all_errors.append(error_data)
    if plot:
        plot_avg_err(all_errors, dataset)
        plot_avg_err_prob(all_errors, dataset)

    all_errors = []
    for i_try_simulate in range(10):
        state_fp = f"output/simulation/{dataset}/results_{i_try_simulate}.h5"
        error_fp = f"output/error/{dataset}/error_{i_try_simulate}_0.json"
        try:
            with open(error_fp, "r") as f:
                error_data = json.load(f)
        except FileNotFoundError:
            error_data = entry.error_estimate(state_fp, data_fp, opt_results)
            with open(error_fp, "w") as f:
                json.dump(error_data, f)
        all_errors.append(error_data)

    if plot:
        plot_avg_bias(all_errors, n_total_papers, dataset)
        plot_avg_err(all_errors, dataset)
        plot_avg_err_prob(all_errors, dataset)


if __name__ == "__main__":
    try:
        dataset = sys.argv[1]
    except IndexError:
        dataset = "ptsd"
    simulate_error(dataset, plot=True)
