# Simulate a dyngen dataset (expression counts + ground-truth GRN) for TopGAE.
#
# Usage (from experiments/simulations/):
#   Rscript simulate_dyngen.R <backbone>
# where <backbone> is one of: bf, branch, conv, cycle, linear, tf
# (bifurcating, branching, converging, cycling, linear, trifurcating; Fig. 2A).
# Default: cycle. Run once per backbone to reproduce all six datasets.
#
# Outputs (read by run_dyngen.py):
#   dyngen_data/dyngen_<backbone>_mat.csv   counts, cells x genes (first column = cell IDs)
#   dyngen_data/dyngen_<backbone>_net.csv   ground-truth network ('from', 'to', ...)
#   dyngen_<backbone>_supp.pdf, dyngen_<backbone>_main.pdf   diagnostic plots

library(utils)
library(tools)
library(ggplot2)
library(dynplot)
library(dyno)
library(Matrix)

# Optional: where dyngen caches downloaded files (defaults to dyngen's own location).
# options(dyngen_download_cache_dir = "/path/to/dyngen/cache")

library(tidyverse)
library(dyngen)

backbones <- list(
  bf     = backbone_bifurcating,
  branch = backbone_branching,
  conv   = backbone_converging,
  cycle  = backbone_cycle,
  linear = backbone_linear,
  tf     = backbone_trifurcating
)

args <- commandArgs(trailingOnly = TRUE)
name <- if (length(args) >= 1) args[1] else "cycle"
if (!name %in% names(backbones)) {
  stop("Unknown backbone '", name, "'. Choose one of: ", paste(names(backbones), collapse = ", "))
}
out_dir <- "dyngen_data"
dir.create(out_dir, showWarnings = FALSE)

set.seed(11)

backbone <- backbones[[name]]()

init <- initialise_model(
  backbone = backbone,
  num_cells = 1000,
  num_tfs = nrow(backbone$module_info),
  num_targets = 500,
  num_hks = 500,
  verbose = FALSE
)
out <- generate_dataset(init, format = "dyno", make_plots = TRUE)

ggsave(paste0("dyngen_", name, "_supp.pdf"), plot = out$plot, width = 30, height = 35, units = "in")

dataset <- out$dataset
model <- out$model
traj <- plot_dimred(dataset)
ggsave(paste0("dyngen_", name, "_main.pdf"), plot = traj, width = 5, height = 5, units = "in")

net_df <- model[["feature_network"]]
mat_df <- as.data.frame(as.matrix(dataset[["counts"]]))

write.csv(mat_df, file.path(out_dir, paste0("dyngen_", name, "_mat.csv")))
write.csv(net_df, file.path(out_dir, paste0("dyngen_", name, "_net.csv")))
