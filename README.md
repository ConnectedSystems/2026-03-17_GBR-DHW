# 2026-03-17_GBR-DHW

Supporting analysis to identify DHW trajectories of interest.

## Setup

All instructions here assume the current directory is the project root.

Place required ADRIA Domain dataset into `data`.
Note: this is really only used for the geospatial data - analysis could be simplified to
only require the [Canonical Reefs](https://github.com/gbrrestoration/canonical-reefs)
dataset.

Create a folder and place NetCDF files into `jb_dhw_mcb`

Initialize project to install all dependencies.

```julia
$ julia --project=.

] instantiate
```

## Running analysis

```julia
include("JB_simple_classify_dhw_trajectories.jl")
```

Results will be placed in a folder called `MCB_traj_classes`.