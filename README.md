# Batch PyTOM template matching and candidate extraction from AreTomo3 outputs

`batch_pytom_aretomo3.py` submits [pytom-match-pick](https://github.com/SBC-Utrecht/pytom-match-pick) jobs to a SLURM cluster. It supports **two workflows**:

1. **Template matching** (`--workflow match`, default): prepare per-tomogram tilt, exposure, and defocus metadata from AreTomo3 outputs and submit `pytom_match_template.py`.
2. **Particle candidate extraction** (`--workflow extract`): find completed matching `*_job.json` files and submit a SLURM array running `pytom_extract_candidates.py`, **one JSON and one GPU per array task**.

Use `--dry-run` to generate submission scripts without calling `sbatch`.

## 1. Template matching

### What the wrapper reads

For each selected `<prefix>.mrc` in `--aretomo-dir`, the wrapper:

- Uses `<prefix>_Imod/<prefix>_st.tlt` (or `<prefix>.tlt` in the same directory) for the raw tilt list.
- Prefers aligned tilt angles from `<prefix>.aln` when available (the TILT value in column 10 of the global alignment records).
- Excludes frames listed in IMOD `tilt.com` `EXCLUDELIST` and AreTomo3 `.aln` `DarkFrame` records.
- Reads defocus from `<prefix>_CTF.txt` when present, falling back to supported IMOD CTF text files; averages the two defocus columns and converts Å to µm. Defocus is optional unless `--tomogram-ctf-model` is requested.
- Computes cumulative electron exposure in acquisition order from `<prefix>_Imod/<prefix>_order_list.csv` and `--dose` (e⁻/Å² per tilt).
- Optionally adds a tomogram boundary mask from `--bmask-dir/<prefix>.mrc`.

By default, all eligible tomograms are processed. Use `--include` and `--exclude` for wildcard prefix filtering.

### Example: one SLURM array (default)

```bash
python batch_pytom_aretomo3.py \
  --workflow match \
  --mode array \
  -i /path/to/aretomo3/output \
  -d submission \
  -t /path/to/template.mrc \
  -m /path/to/mask.mrc \
  -g 0 \
  --voxel-size-angstrom 7.64 \
  --dose 2 \
  --particle-diameter 140 \
  --array-max-parallel 8 \
  --dry-run
```

Remove `--dry-run` to submit. For **one separate SLURM job per tomogram**, use `--mode per-tomo` instead. `--mode` controls *template-matching submission*, not the choice of workflow.

You can use `--angular-search 10` instead of `--particle-diameter 140` for matching, but not both. Optional parameters include `-s 2 2 1`, `--per-tilt-weighting`, `--non-spherical-mask`, `--tomogram-ctf-model phase-flip`, `-r`, and `--rng-seed 69`.

### Template-matching output

With default `--mode array`:

```text
submission/
├── prefixes.txt
├── submit_array.sh
├── Position_1/
│   ├── Position_1.tlt
│   ├── Position_1_defocus.txt       # only if defocus metadata available
│   └── Position_1_exposure.txt
├── Position_2/
└── Position_3/
```

With `--mode per-tomo`, each tomogram subfolder also contains `submit_<prefix>.sh`. After a successful `pytom_match_template.py` run, the matching output includes a `*_job.json` file that extraction can use.

## 2. Particle candidate extraction

After template matching finishes, run the same wrapper in extraction mode. The wrapper recursively discovers `*_job.json` files under `--output-dir`, sorts their paths, and assigns **one JSON per SLURM array task**. No AreTomo3 directory, template, mask, dose, voxel size, or `-g` argument is required for this mode. Optionally, `--bmask-dir` can be supplied here, overriding any previous tomogram masks used during template matching.

### Example: defaults

```bash
python batch_pytom_aretomo3.py \
  --workflow extract \
  -d submission \
  --dry-run
```

The JSON filename shown is illustrative; the wrapper uses the actual discovered filename and absolute path.

### Example: customize extraction

```bash
python batch_pytom_aretomo3.py \
  --workflow extract \
  -d submission \
  --particle-diameter 300 \
  -n 1000 \
  -c 0.1 \
  --relion5-compat \
  --array-max-parallel 8 \
  --extract-extra-args '--some-option value --another-flag' \
  --dry-run
```

Replace the placeholder options in `--extract-extra-args` with flags accepted by your installed version of `pytom_extract_candidates.py`. Shell-style quoting is supported. Do not repeat wrapper-managed options such as `-j`, `--particle-diameter`, `-n`, `-c`, or `--relion5-compat` in the extra-args string. Use `--no-relion5-compat` to turn off RELION5 compatibility.

**GPU enforcement:** extraction always generates `#SBATCH --gres=gpu:1`, even if `--gres` is set to another value. Extraction always uses a SLURM array; `--mode per-tomo` does not change it. `--array-max-parallel` limits simultaneous array tasks (default: 8).

### Extraction output

The wrapper generates these files in `submission/`:

```text
submission/
├── extract_jobs.txt              # sorted absolute paths to matching JSON files
├── submit_extract_array.sh       # SLURM array script
├── Position_1/
│   └── ..._job.json              # pre-existing matching result
└── Position_2/
    └── ..._job.json
```

When submitted, task logs are named `pytom_extract_<jobid>_<taskid>.out`. Candidate outputs are written by PyTOM according to its own extraction behavior.

`--include` and `--exclude` also work in extraction mode: patterns match either the JSON's parent folder name or its filename without `_job.json`.

## 3. Flag summary

### Workflow and common options

| Flag | Description | Default |
|---|---|---|
| `--workflow {match,extract}` | Select template matching or candidate extraction | `match` |
| `-d, --output-dir` | Matching output root; extraction scans it for `*_job.json` | `submission` |
| `--include` / `--exclude` | Filter tomogram prefixes or extraction job JSONs using wildcards | all / none |
| `--dry-run` | Generate scripts but do not submit | off |
| `--array-max-parallel`, `--max-parallel` | Maximum simultaneous array tasks | `8` |

### Template matching: required inputs

| Flag | Description | Default |
|---|---|---|
| `-i, --aretomo-dir` | Directory of AreTomo3 outputs | required for `match` |
| `-t, --template` | Template MRC | required for `match` |
| `-m, --mask` | Template mask MRC | required for `match` |
| `-g, --gpu-ids` | GPU IDs passed to `pytom_match_template.py` | required for `match` |
| `--voxel-size-angstrom` | Voxel size (Å) | required for `match` |
| `--dose` | Dose per tilt (e⁻/Å²) | required for `match` |
| `--particle-diameter` **or** `--angular-search` | Crowther sampling diameter (Å), or angular search override | one required for `match` |

### Template matching: other options

| Flag | Description | Default |
|---|---|---|
| `--mode {array,per-tomo}` | One matching array or individual matching jobs | `array` |
| `--bmask-dir` | Boundary masks named `<prefix>.mrc` | none |
| `-s, --volume-split X Y Z` | Split matching search volume | none |
| `--search-x START END`, `--search-y START END`, `--search-z START END` | Limit search ranges | none |
| `--non-spherical-mask` | Use non-spherical template mask | off |
| `--z-axis-rotational-symmetry` | Z-axis symmetry order | none |
| `--tomogram-ctf-model` | Reconstruction CTF model (`phase-flip`) | none |
| `-r, --random-phase-correction` | Random phase correction | off |
| `--rng-seed` | Random seed (passed with `-r`) | `69` |
| `--half-precision` | Float16 output | off |
| `--per-tilt-weighting` | Per-tilt weighting | off |
| `--low-pass`, `--high-pass` | Filtering cutoffs in Å | none |
| `--phase-shift` | Phase shift in degrees | none |
| `--defocus-handedness` | Defocus gradient handedness: `-1`, `0`, or `1` | none |
| `--spectral-whitening` | Spectral whitening | off |
| `--relion5-tomograms-star` | RELION5 tomograms STAR path | none |
| `--amplitude-contrast` | Amplitude contrast | `0.07` |
| `--spherical-aberration` | Spherical aberration (mm) | `2.7` |
| `--voltage` | Accelerating voltage (kV) | `300` |
| `--log` | PyTOM logging level (`info` or `debug`) | none |

### Candidate extraction options

| Flag | Description | Default |
|---|---|---|
| `--particle-diameter` | Particle diameter (Å) for extraction | `300` in `extract` |
| `-n, --n-candidates` | Maximum candidates per JSON | `1000` |
| `-c, --cutoff` | Candidate score cutoff | `0.1` |
| `--relion5-compat` | RELION5-compatible extraction | enabled |
| `--no-relion5-compat` | Disable RELION5 compatibility | — |
| `--extract-extra-args` | Additional extraction CLI arguments as a quoted string | empty |

### SLURM settings

| Flag | Description | Default |
|---|---|---|
| `--partition` | Partition | `emgpu` |
| `--qos` | Quality of Service | `emgpu` |
| `--ntasks` | Number of tasks | `1` |
| `--nodes` | Number of nodes | `1` |
| `--ntasks-per-node` | Tasks per node | `1` |
| `--cpus-per-task` | CPUs per task | `4` |
| `--gres` | GPU request (extraction overrides with `gpu:1`) | `gpu:1` |
| `--mem` | Memory in GB | `128` |
| `--time` | Time limit | `05:00:00` |
| `--mail-type` | Mail notifications | `none` |
| `--exclude-nodes` | Nodes to exclude | none |
| `--include-nodes` | Nodes to request | none |

## Full help

```bash
python batch_pytom_aretomo3.py -h
```

The wrapper generates scripts and submits jobs to SLURM; it does not wait for template matching to finish before extraction. Run the extraction workflow **after matching has completed**.
