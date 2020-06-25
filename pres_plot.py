#!/usr/bin/env python

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import norm

from asreview.state import open_state
from asreview.models import get_model
from asreview.balance_strategies import get_balance_model
from asreview.feature_extraction import get_feature_model
from asreview import ASReviewData


def corrected_proba(X, y, train_one_idx, train_zero_idx, n_sample=10):
    cor_proba = []
    for _ in range(n_sample):
        if len(train_one_idx) == 1:
            new_train_idx = np.append(train_one_idx, train_zero_idx)
            X_train, y_train = balance_model.sample(X, y, new_train_idx, {})
            model.fit(X_train, y_train)
            correct_proba = model.predict_proba(X[train_one_idx])[0, 1]
            cor_proba.append(correct_proba)
            continue

        for i_rel_train in range(len(train_one_idx)):
            new_train_idx = np.append(np.delete(train_one_idx, i_rel_train),
                                      train_zero_idx)
            X_train, y_train = balance_model.sample(X, y, new_train_idx, {})
            model.fit(X_train, y_train)
            correct_proba = model.predict_proba(X[train_one_idx[i_rel_train]])[0, 1]
            cor_proba.append(correct_proba)

    return np.array(cor_proba)


dataset_name = "ace"

data_fp = Path("data", dataset_name + ".csv")
state_fp = Path("output", dataset_name + ".h5")
optimization_fp = Path("output", dataset_name + ".h5")

query_i = 5

feature_model = get_feature_model("tfidf")

as_data = ASReviewData.from_file(data_fp)
X = feature_model.fit_transform(
    as_data.texts, as_data.headings, as_data.bodies, as_data.keywords)

with open_state(state_fp) as state:
    train_idx = state.get("train_idx", query_i)
    pool_idx = state.get("pool_idx", query_i)

y = as_data.labels
train_one_idx = train_idx[np.where(y[train_idx] == 1)[0]]
train_zero_idx = train_idx[np.where(y[train_idx] == 0)[0]]
pool_one_idx = pool_idx[np.where(y[pool_idx] == 1)[0]]
model = get_model("nb")
balance_model = get_balance_model("double")
X_train, y_train = balance_model.sample(X, y, train_idx, {})
model.fit(X_train, y_train)
proba = model.predict_proba(X)[:, 1]
df_all = -np.log(1/proba-1)
proba_train_one_cor = corrected_proba(X, y, train_one_idx, train_zero_idx)
df_train_one_cor = -np.log(1/proba_train_one_cor-1)

n_bins = 50
df_range = (np.min(df_all), np.max(df_all))

hist_all, bin_edges = np.histogram(df_all, bins=n_bins, range=df_range,
                                   density=False)
hist_train_one, _ = np.histogram(df_all[train_one_idx], bins=n_bins, range=df_range,
                                 density=False)
hist_train_zero, _ = np.histogram(df_all[train_zero_idx], bins=n_bins, range=df_range,
                                  density=False)
hist_pool_one, _ = np.histogram(df_all[pool_one_idx], bins=n_bins, range=df_range,
                                density=False)
hist_train_one_cor, _ = np.histogram(df_train_one_cor, bins=n_bins, range=df_range,
                                     density=False)
hist_pool, _ = np.histogram(df_all[pool_idx], bins=n_bins, range=df_range, density=False)

prob_all, proba_edges = np.histogram(
    proba, bins=n_bins, range=(0, 1), density=False)

one_dist = norm.fit(df_all[train_one_idx])
one_dist_cor = norm.fit(df_train_one_cor)

hist_train = (hist_train_zero + hist_train_one_cor/10)
perc_train = (hist_train + 0.0000001)/(hist_train + hist_pool + 0.0000001)

x_bin = (bin_edges[1:]+bin_edges[:-1])/2
d_bin = bin_edges[1] - bin_edges[0]

f_train = 1/(len(train_one_idx)+len(pool_one_idx))/d_bin
f_pool = 1/(len(train_one_idx)+len(pool_one_idx))/d_bin

plt.grid()
plt.xlabel("Model 'probability'")
plt.ylabel("Count")
plt.plot((proba_edges[1:]+proba_edges[:-1])/2, prob_all)
plt.savefig("pics/prob_hist.png")
plt.close()
# plt.show()

plt.xlabel("Model decision function")
plt.ylabel("Count")
plt.grid()
plt.plot(x_bin, hist_all)
plt.savefig("pics/df_hist.png")
plt.close()
# plt.show()

plt.grid()
plt.plot(x_bin, f_train*hist_train_one, color="green", label="labeled included")
plt.plot(x_bin, f_pool*hist_pool_one, color="red", label="unlabeled included")
plt.plot(x_bin, hist_pool/len(pool_idx)/d_bin, color="black", label="unlabeled")
plt.plot(bin_edges, norm(*one_dist).pdf(bin_edges), linestyle=":", color="green")
plt.legend()
plt.savefig("pics/df_dist.png")
plt.close()
# plt.show()

plt.grid()
plt.plot(x_bin, hist_train_one/len(train_one_idx)/d_bin,
         color="red", label="labeled 1 uncorrected")
plt.plot(x_bin, norm(*one_dist).pdf(x_bin), linestyle=":", color="red")
plt.plot(x_bin, hist_train_one_cor/len(df_train_one_cor)/d_bin,
         color="green", label="labeled 1 corrected")
plt.plot(x_bin, norm(*one_dist_cor).pdf(x_bin), linestyle=":", color="green")
plt.xlabel("Decision function")
plt.ylabel("Probability density")
plt.legend()
plt.savefig("pics/label_corrected.png")
plt.close()
# plt.show()

plt.grid()
plt.plot(x_bin, hist_train_one_cor/len(df_train_one_cor)/d_bin,
         color="green", label="labeled included corrected")
plt.plot(x_bin, norm(*one_dist_cor).pdf(x_bin), linestyle=":", color="green")
plt.plot(x_bin, perc_train)
plt.plot(x_bin, hist_pool/len(pool_idx)/d_bin, color="black", label="unlabeled")
plt.plot(x_bin, f_pool*hist_pool_one, color="red", label="unlabeled included")
plt.xlabel("Decision function")
plt.legend(loc="upper right")
plt.savefig("pics/percent_found.png")
plt.close()
# plt.show()
