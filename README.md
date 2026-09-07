# Capacity-Audit Reproducibility Package

Minimal package reproducing the capacity-availability audit of a
regional construction-and-demolition waste facility-location system:
the capacity ladder (288 solves), its holding-regret analysis with
bootstrap intervals and expansion-cost trade-off, and the
enforcement-by-capacity grid (144 solves).

## Layout

```
data/raw/            frozen input snapshot (read-only)
  cdw_estimation_1990_2022.csv   city-year C&D generation panel
  facility_list.csv              candidate-facility parameterization
src/capacity_audit/         core library (instance schema, covariance models,
                     scenario generation, mean-CVaR SAA solver, recourse LP)
experiments/
  common.py          shared setup: instance builder, four covariance
                     specifications, design constants
  run_ladder.py      Experiment 1 -> results/ladder.json
  analyze_ladder.py  analysis     -> results/ladder_analysis.json
  run_enforcement.py Experiment 2 -> results/enforcement.json
results/             output directory (created on first run)
```

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `scipy`
- IBM ILOG CPLEX with `docplex` (tested with CPLEX 22.1.1 /
  docplex 2.24; the extensive-form solves need a local CPLEX
  installation)

Install Python dependencies with:

```
pip install -r requirements.txt
```

## Running

From the repository root:

```
python experiments/run_ladder.py        # ~1 h on one core
python experiments/analyze_ladder.py    # ~10 min (scenario re-evaluation)
python experiments/run_enforcement.py   # ~1 h on one core
```

All seeds, scenario counts, and grid points are fixed inside the
scripts; outputs are deterministic given the same library versions.

## Notes

- Facility capacities and fixed costs in `facility_list.csv` are a
  reproducible parameterization from regional planning parameters, not
  observed facility accounts; the demand panel is a statistical
  estimate from public construction-activity statistics.
- The reference layout held fixed throughout (16 of 33 facilities) is
  listed in `experiments/common.py` (`CERTIFIED`); it is the optimal
  layout at the primary setting and coincides across the four
  covariance specifications.
