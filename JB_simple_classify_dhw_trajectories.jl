"""
JB_simple_classify_dhw_trajectories.jl

For each (GCM × region × albedo × deployment scale × MCB scenario),
the regional mean DHW trajectory was characterised by:

  slope          — OLS slope fit to a rolling mean of the spatial mean DHW
                   trajectory (RollingFunctions.jl).  Captured the sustained
                   rate of DHW change, smoothing over year-to-year noise.
  variability    — standard deviation of the spatial mean DHW trajectory
                   around the rolling mean (overall noise level).
  variance_slope — OLS slope of the rolling standard deviation over time.
                   A positive value indicated variability was growing as DHW
                   rose, even if the overall noise level was modest.
  lower95_DHW    — 5th  percentile across all reef-years in the region
  upper95_DHW    — 95th percentile across all reef-years in the region
  max_lower95    — highest per-reef 5th  percentile (most consistently
                   stressed reef — high values indicate no low-stress refuge)
  min_upper95    — lowest  per-reef 95th percentile (least extreme reef —
                   near-zero values indicate some reefs experience no stress)

Each series was then classified into one of six classes:
  Slow Rise  / Low Var          |   Rapid Rise / Low Var
  Slow Rise  / High Overall Var |   Rapid Rise / High Overall Var
  Slow Rise  / Increasing Var   |   Rapid Rise / Increasing Var

Trend threshold: median slope across all PDP regional series.
Variance class (priority-ordered):
  High Overall Var — overall variability above median
  Increasing Var   — overall variability below median, but variance slope
                     above median (calm now, getting noisier over time)
  Low Var          — below median on both measures

Thresholds were derived from PDP regional series only (GBR excluded).
GBR was subsequently assigned the modal class of its constituent PDP
regions, ensuring consistency between scales.

── Outputs ───────────────────────────────────────────────────────────────────

  trajectory_classifications.csv    — one row per (file × scenario × region)
  summary_trajectories.png          — mean series for all spatial areas, coloured by class
  <file_stem>_<var>_regional.png    — means for the spatial areas for that file/scenario

── Use of AI disclosure ──────────────────────────────────────────────────────

The analysis process and steps were conceptualized by T. Iwanaga
An initial script for analysis was written by hand and iterated on with Claude Sonnet 4.6
All AI generated code was reviewed and manual adjustments were made where necessary.
Code style and format were left as-is where modified by AI.

"""

using NCDatasets
using CairoMakie
using Statistics
using StatsBase
using RollingFunctions
using DataFrames
using CSV
using ADRIA
using GeoDataFrames

# ── User configuration ─────────────────────────────────────────────────────────

const DATA_DIR = "./jb_dhw_mcb"
const OUTPUT_DIR = "./MCB_traj_classes"
const ADRIA_DOMAIN_PATH = "./data/GBR_HighResCoralStress_2026-03-17_v080"
const REGION_GPKG = "./data/PDP_reefs.gpkg"

const ADRIA_ID_COLUMN = :UNIQUE_ID
const PDP_REGION_COLUMN = :PDP_region

const YEAR_FLOOR = 2025
const ROLLING_WINDOW = 10    # years — rolling mean window for slope estimation
const TARGET_SSP = ["ssp245"]
const TARGET_ALBEDO = ["albedo-02", "albedo-03"]
const MCB_VARS = ["dhw_max_0", "dhw_max_50", "dhw_max_100", "dhw_max_150"]

const MCB_LABELS = Dict(
    "dhw_max_0" => "No MCB (0 days)",
    "dhw_max_50" => "MCB — 50 days",
    "dhw_max_100" => "MCB — 100 days",
    "dhw_max_150" => "MCB — 150 days"
)

const CLASS_COLOURS = Dict{String,RGBAf}(
    "Slow Rise / Low Var" => RGBAf(0.27, 0.51, 0.71, 1.0),  # steel blue
    "Slow Rise / High Overall Var" => RGBAf(0.12, 0.56, 1.00, 1.0),  # dodger blue
    "Slow Rise / Increasing Var" => RGBAf(0.00, 0.75, 0.75, 1.0),  # teal
    "Rapid Rise / Low Var" => RGBAf(1.00, 0.55, 0.00, 1.0),  # dark orange
    "Rapid Rise / High Overall Var" => RGBAf(1.00, 0.39, 0.28, 1.0),  # tomato
    "Rapid Rise / Increasing Var" => RGBAf(0.70, 0.13, 0.13, 1.0),  # fire brick
    "Unclassified" => RGBAf(0.50, 0.50, 0.50, 1.0)
)

# ── Filename metadata parsing ──────────────────────────────────────────────────

"""
    parse_filename(stem::String)::NamedTuple

Extract GCM name, deployment scale, and albedo value from a NetCDF file stem.

Expected pattern (flexible):
  CoralSea_GBR_<GCM>_<ssp>_<run>_dhw_<years>-reefs-MCB-<deployment>-albedo-<XX>.nc
"""
function parse_filename(stem::String)::NamedTuple
    gcm = let m = match(r"GBR_([^_]+(?:[-_][^_]+)*?)_ssp", stem)
        isnothing(m) ? "unknown" : m.captures[1]
    end
    deployment = let m = match(r"MCB-([A-Za-z]+)-albedo", stem)
        isnothing(m) ? "unknown" : m.captures[1]
    end
    albedo = let m = match(r"albedo-(\d+)", stem)
        isnothing(m) ? "unknown" : "albedo-" * m.captures[1]
    end
    return (gcm=gcm, deployment=deployment, albedo=albedo)
end

# ── Feature extraction ─────────────────────────────────────────────────────────

"""
    ols_slope(y) → Float64

OLS slope of `y` over an implicit unit-spaced x-axis [1, …, n].
Equivalent to cov(x, y) / var(x) with no external dependency.
"""
function ols_slope(y::AbstractVector{<:Real})::Float64
    x = collect(1.0:length(y))
    return cov(x, y) / var(x)
end

"""
    series_features(μ_traj, subset, window) → NamedTuple

Compute all characterisation statistics for one (region × file × scenario):
  - slope          : OLS slope of the rollmean of `μ_traj` (RollingFunctions.jl)
  - variability    : std of `μ_traj` around its rollmean (overall noise level)
  - variance_slope : OLS slope of the rollstd of `μ_traj` — positive means
                     variability is growing over time, negative means stabilising
  - lower95_DHW    : 5th  percentile across all reef-years in `subset`
  - upper95_DHW    : 95th percentile across all reef-years in `subset`
  - max_lower95    : maximum per-reef 5th  percentile
  - min_upper95    : minimum per-reef 95th percentile
"""
function series_features(
    μ_traj::Vector{Float64},
    subset::Matrix{Float64},
    window::Int
)::NamedTuple
    rm = rollmean(μ_traj, window)
    slope = ols_slope(rm)
    variability = std(μ_traj[1:length(rm)] .- rm)
    variance_slope = ols_slope(rollstd(μ_traj, window))

    flat = filter(!isnan, vec(subset))
    lo95 = isempty(flat) ? NaN : quantile(flat, 0.05)
    hi95 = isempty(flat) ? NaN : quantile(flat, 0.95)

    per_reef_lo = [
        begin
            v = filter(!isnan, subset[i, :])
            isempty(v) ? NaN : quantile(v, 0.05)
        end
        for i in axes(subset, 1)
    ]
    per_reef_hi = [
        begin
            v = filter(!isnan, subset[i, :])
            isempty(v) ? NaN : quantile(v, 0.95)
        end
        for i in axes(subset, 1)
    ]

    return (
        slope=slope,
        variability=variability,
        variance_slope=variance_slope,
        lower95_DHW=lo95,
        upper95_DHW=hi95,
        max_lower95=isempty(filter(!isnan, per_reef_lo)) ? NaN :
                    maximum(filter(!isnan, per_reef_lo)),
        min_upper95=isempty(filter(!isnan, per_reef_hi)) ? NaN :
                    minimum(filter(!isnan, per_reef_hi))
    )
end

# ── Classification ─────────────────────────────────────────────────────────────

"""
    classify_rows!(df) → DataFrame

Add a `traj_class` column to `df` in-place.

Thresholds are derived from PDP regional rows only (region != "GBR") and then
applied uniformly to all rows.  GBR rows are subsequently overwritten with the
modal class of their constituent PDP regions for that (file, scenario).

Trend dimension (median split on slope):
  Slow Rise | Rapid Rise

Variance dimension (three-way split on variability and variance_slope):
  Low Var          — below median on both overall variability and variance slope
  High Overall Var — above median overall variability, regardless of trend in var
  Increasing Var   — below median overall variability but above median variance
                     slope (i.e. starts calm but is getting noisier over time)
"""
function classify_rows!(df::DataFrame)::DataFrame
    pdp_rows = df[df.region .!= "GBR", :]
    slope_med = median(pdp_rows.slope)
    var_med = median(pdp_rows.variability)
    vslope_med = median(pdp_rows.variance_slope)

    function var_class(v::Float64, vs::Float64)::String
        if v >= var_med
            "High Overall Var"
        elseif vs >= vslope_med
            "Increasing Var"
        else
            "Low Var"
        end
    end

    df[!, :traj_class] = map(df.slope, df.variability, df.variance_slope) do s, v, vs
        trend = s >= slope_med ? "Rapid Rise" : "Slow Rise"
        "$trend / $(var_class(v, vs))"
    end

    # Overwrite GBR rows with modal class of constituent PDP regions
    gbr_mask = df.region .== "GBR"
    for gbr_row in eachrow(df[gbr_mask, :])
        region_rows = df[
            (df.file .== gbr_row.file) .& (df.scenario .== gbr_row.scenario) .& (df.region .!= "GBR"),
            :
        ]
        isempty(region_rows) && continue

        counts = countmap(region_rows.traj_class)
        max_count = maximum(values(counts))
        candidates = [k for (k, v) in counts if v == max_count]

        winner = if length(candidates) == 1
            candidates[1]
        else
            best = argmax([
                mean(region_rows[region_rows.traj_class .== c, :upper95_DHW])
                for c in candidates
            ])
            candidates[best]
        end

        df[
            (df.file .== gbr_row.file) .& (df.scenario .== gbr_row.scenario) .& (df.region .== "GBR"),
            :traj_class
        ] .= winner
    end

    return df
end

# ── Spatial region assignment ──────────────────────────────────────────────────

function assign_regions(
    loc_data::DataFrame,
    gpkg_path::String,
    region_col::Symbol
)::Vector{Union{String,Missing}}
    region_gdf = GeoDataFrames.read(gpkg_path)
    region_lookup = region_gdf[:, [ADRIA_ID_COLUMN, region_col]]

    loc_data.PDP_region .= ""
    for reg in unique(region_lookup.PDP_region)
        target_ids = region_lookup[region_lookup.PDP_region .== reg, :UNIQUE_ID]
        loc_data[loc_data.UNIQUE_ID .∈ Ref(target_ids), region_col] .= reg
    end

    loc_data.PDP_region = replace(loc_data.PDP_region, "" => missing)

    return loc_data.PDP_region
end

# ── Plotting helpers ───────────────────────────────────────────────────────────

class_colour(label::String) = get(CLASS_COLOURS, label, CLASS_COLOURS["Unclassified"])

function plot_classified_series(
    years::Vector{Int},
    series_mat::Matrix{Float64},
    labels::Vector{String},
    series_names::Vector{String},
    title::String;
    fig_size=(1600, 700)
)
    fig = Figure(; size=fig_size)
    ax = Axis(fig[1, 1]; title=title, xlabel="Year", ylabel="Mean DHW (°C-weeks)")

    legend_handles = Any[]
    legend_texts = String[]

    for (j, (lbl, name)) in enumerate(zip(labels, series_names))
        col = class_colour(lbl)
        l = lines!(ax, years, series_mat[:, j]; color=col, linewidth=1.8)
        if lbl ∉ legend_texts
            push!(legend_handles, l)
            push!(legend_texts, lbl)
        end
    end

    Legend(fig[1, 2], legend_handles, legend_texts; labelsize=11, framevisible=false)
    return fig
end

function plot_regional_series(
    years::Vector{Int},
    region_data::Matrix{Float64},
    region_labels::Vector{String},
    region_names::Vector{String},
    title::String
)
    fig = Figure(; size=(1200, 520))
    ax = Axis(fig[1, 1]; title=title, xlabel="Year", ylabel="Mean DHW (°C-weeks)")

    legend_handles = Any[]
    legend_texts = String[]

    for (j, (lbl, name)) in enumerate(zip(region_labels, region_names))
        col = class_colour(lbl)
        l = lines!(ax, years, region_data[:, j]; color=col, linewidth=2.2)
        push!(legend_handles, l)
        push!(legend_texts, "$name  [$lbl]")
    end

    Legend(fig[1, 2], legend_handles, legend_texts; labelsize=11, framevisible=false)
    return fig
end

# ── Initialise ─────────────────────────────────────────────────────────────────

mkpath(OUTPUT_DIR)

@info "Loading ADRIA domain from $ADRIA_DOMAIN_PATH"
dom = ADRIA.load_domain(ADRIA_DOMAIN_PATH, "45")

adria_ids = dom.loc_data[:, ADRIA_ID_COLUMN]
n_adria_locs = length(adria_ids)
adria_idx_map = Dict(id => i for (i, id) in enumerate(adria_ids))

@info "Assigning reef locations to regions via $REGION_GPKG"
reef_regions = assign_regions(dom.loc_data, REGION_GPKG, PDP_REGION_COLUMN)
unique_regions = sort(collect(Set(skipmissing(reef_regions))))
n_outside = sum(ismissing.(reef_regions))

@info "  PDP regions: $(join(unique_regions, ", "))"
@info "  $n_outside / $n_adria_locs reefs outside all region polygons"

all_region_names = ["GBR"; unique_regions]

region_masks = Dict{String,BitVector}("GBR" => trues(n_adria_locs))
for r in unique_regions
    region_masks[r] = coalesce.(reef_regions .=== r, false)
end

# ── Collection pass ────────────────────────────────────────────────────────────

nc_files = sort(filter(f -> endswith(f, ".nc"), readdir(DATA_DIR; join=true)))
nc_files = filter(f -> any(occursin.(TARGET_SSP, f)), nc_files)
nc_files = filter(f -> any(occursin.(TARGET_ALBEDO, f)), nc_files)
@info "Found $(length(nc_files)) NetCDF file(s) matching filters"

rows = NamedTuple[]    # one element per (file × var × region)
series_cols = Vector{Vector{Float64}}()
ref_years = nothing

for nc_path in nc_files
    file_stem = splitext(basename(nc_path))[1]
    file_meta = parse_filename(file_stem)
    @info "━━ Collecting: $file_stem"

    NCDataset(nc_path, "r") do ds
        all_years = Int.(ds["year_dhw"][:])
        nc_ids = string.(Int64.(ds["UNIQUE_ID"][:]))

        year_mask = YEAR_FLOOR == -Inf ? trues(length(all_years)) : all_years .>= YEAR_FLOOR
        years = all_years[year_mask]

        global ref_years
        if isnothing(ref_years)
            ref_years = years
        elseif ref_years != years
            @warn "  Year axis mismatch — skipping $file_stem"
            return nothing
        end

        nc_to_adria = [get(adria_idx_map, id, 0) for id in nc_ids]
        n_unmatched = sum(nc_to_adria .== 0)
        n_unmatched > 0 &&
            @warn "  $n_unmatched / $(length(nc_ids)) reef IDs not in ADRIA domain"

        for var in MCB_VARS
            raw = Float64.(ds[var][:, :])
            data_yr_reef = size(raw, 1) == length(all_years) ? raw : raw'
            data_yr_reef = data_yr_reef[year_mask, :]

            adria_grid = fill(NaN, n_adria_locs, length(years))
            for (nc_i, adria_i) in enumerate(nc_to_adria)
                adria_i == 0 && continue
                adria_grid[adria_i, :] = data_yr_reef[:, nc_i]
            end

            for region in all_region_names
                subset = adria_grid[region_masks[region], :]
                μ_traj = [mean(filter(!isnan, subset[:, t])) for t in axes(subset, 2)]
                feats = series_features(μ_traj, subset, ROLLING_WINDOW)

                push!(series_cols, μ_traj)
                push!(
                    rows,
                    (
                        file=file_stem,
                        gcm=file_meta.gcm,
                        deployment=file_meta.deployment,
                        albedo=file_meta.albedo,
                        scenario=MCB_LABELS[var],
                        var=var,
                        region=region,
                        slope=feats.slope,
                        variability=feats.variability,
                        variance_slope=feats.variance_slope,
                        lower95_DHW=feats.lower95_DHW,
                        upper95_DHW=feats.upper95_DHW,
                        max_lower95=feats.max_lower95,
                        min_upper95=feats.min_upper95
                    )
                )
            end

            @info "  ├─ $(MCB_LABELS[var])  ✓"
        end
    end
end

series_mat = hcat(series_cols...)
n_series = length(rows)
@info "Collected $n_series series"

# ── Classify ───────────────────────────────────────────────────────────────────

class_df = classify_rows!(DataFrame(rows))

@info "Class counts: " *
    join(
    [
        "$lbl: $(sum(class_df.traj_class .== lbl))"
        for lbl in sort(unique(class_df.traj_class))
    ], ", ")

csv_path = joinpath(OUTPUT_DIR, "trajectory_classifications.csv")
CSV.write(csv_path, class_df)
@info "Classification CSV → $csv_path"

# ── Figures ────────────────────────────────────────────────────────────────────

series_names = [
    "$(r.gcm) / $(r.deployment) / $(r.albedo) / $(r.scenario) / $(r.region)"
    for r in rows
]
traj_labels = class_df.traj_class

# 1. Summary — all series
fig_summary = plot_classified_series(
    ref_years, series_mat, traj_labels, series_names,
    "DHW Regional Mean Trajectories — All Files × Scenarios × Regions"
)
save(joinpath(OUTPUT_DIR, "summary_trajectories.png"), fig_summary)

# 2. Per-file × var — four regional means
for (file_stem, var) in Iterators.product(unique(class_df.file), MCB_VARS)
    mask = (class_df.file .== file_stem) .& (class_df.var .== var)
    sum(mask) == 0 && continue

    sub = class_df[mask, :]
    row_idx = [findfirst(sub.region .== r) for r in all_region_names]
    valid = .!isnothing.(row_idx)
    row_idx = row_idx[valid]

    reg_data = series_mat[:, findall(mask)[row_idx]]
    reg_labels = traj_labels[findall(mask)[row_idx]]
    reg_names = all_region_names[valid]

    fig = plot_regional_series(
        ref_years, reg_data, reg_labels, reg_names,
        "$file_stem\n$(MCB_LABELS[var]) — Regional Mean DHW Trajectories"
    )
    save(joinpath(OUTPUT_DIR, "$(file_stem)_$(var)_regional.png"), fig)
end

@info "All figures saved → $OUTPUT_DIR"
