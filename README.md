# Batch Submission of `pytom-match-pick` from AreTomo3 Output

This script automates batch submission of [pytom-match-pick](https://github.com/SBC-Utrecht/pytom-match-pick) jobs on an HPC cluster (SLURM) by reading metadata directly from **AreTomo3** outputs. It:

- Extracts **tilt angles** from `IMOD/<prefix>_st.tlt`
- Reads **defocus** (average of defocus1/defocus2 in Å, converted to μm) from `<prefix>_CTF.txt`
- Computes **per-tilt exposure** by cumulatively summing `--dose` in acquisition order (`<prefix>_Imod/<prefix>_order_list.csv`)
- Generates one SLURM submission script per tomogram

and will run on all tomograms/<prefix>.mrc files by default unless specified `--include` or `--exclude`

`--dry-run` will generate the bash scripts without submitting them, allowing for quick sanity checks or manual execution.

---

## Usage
```bash
./batch_pytom_aretomo3.py \
  -i /path/to/aretomo3/output \  # required
  -d submission \                # default: submission
  -t /path/to/template.mrc \    # required
  -m /path/to/mask.mrc \        # required
  -g 0 \                         # required, GPU IDs
  --voxel-size-angstrom 7.64 \  # required
  --dose 2 \                     # required, e-/Å² per tilt
  [--include Position_*] [--exclude Position_5] \  # optional wildcard filtering
  [--angular-search 10] [--particle-diameter 140] \ # either or required
  -s 2 2 1 \                    # optional
  --per-tilt-weighting \       # optional
  --non-spherical-mask \       # optional
  --tomogram-ctf-model phase-flip \ # optional
  -r \                          # optional
  --rng-seed 69 \               # default: 69
  [--dry-run]                    # optional
```

## Flag Summary

### Core flags
| Flag                          | Description                                      | Default     |
|-------------------------------|--------------------------------------------------|-------------|
| `-i, --aretomo-dir`           | AreTomo3 output directory                        | —           |
| `-t, --template`              | Template MRC for matching                        | —           |
| `-m, --mask`                  | Mask MRC for matching                            | —           |
| `-g, --gpu-ids`               | GPU IDs (e.g. `0` or `0 1`)                      | —           |
| `--voxel-size-angstrom`       | Voxel size in Å                                  | —           |
| `--dose`                      | Electron dose per tilt (e‑/Å²)                   | —           |
| `-d, --output-dir`            | Top-level folder for outputs                     | `submission`|
| `--include` / `--exclude`     | Wildcard patterns for prefix filtering           | all / none  |
| `--particle-diameter`          | Particle diameter in Å (Crowther sampling)      | none        |
| `--angular-search`             | Override angular search (float max or `.txt`)   | none        |
| `--dry-run`                   | Generate scripts without submitting              | off         |

### Input Options
| Flag                           | Description                                      | Default     |
|--------------------------------|--------------------------------------------------|-------------|
| `-s, --volume-split`           | Split volume into X Y Z blocks                   | none        |
| `--search-x START END`         | Search range along x-axis                        | none        |
| `--search-y START END`         | Search range along y-axis                        | none        |
| `--search-z START END`         | Search range along z-axis                        | none        |
| `--non-spherical-mask`         | Enable non-spherical mask support                | off         |
| `--tomogram-ctf-model`         | CTF model used (e.g. `phase-flip`)               | none        |
| `-r, --random-phase-correction`| STOPGAP-style random-phase correction            | off         |
| `--half-precision`             | Use float16 output                               | off         |
| `--rng-seed`                   | RNG seed for phase correction                    | `69`        |
| `--per-tilt-weighting`         | Enable per-tilt CTF weighting                    | off         |
| `--low-pass LOW_PASS`          | Low-pass filter cutoff in Å                      | none        |
| `--high-pass HIGH_PASS`        | High-pass filter cutoff in Å                     | none        |
| `--phase-shift PHASE_SHIFT`    | Phase shift in degrees                           | `0.0`       |
| `--defocus-handedness`         | Defocus gradient handedness (`-1,0,1`)           | none        |
| `--spectral-whitening`         | Enable spectral whitening                        | off         |
| `--relion5-tomograms-star`     | Path to RELION5 `tomograms.star` override        | none        |

### Written by default!
| Flag                          | Description                                      | Default     |
|-------------------------------|--------------------------------------------------|-------------|
| `--amplitude-contrast`        | Amplitude contrast fraction                      | `0.07`      |
| `--spherical-aberration`      | Spherical aberration (mm)                        | `2.7`       |
| `--voltage`                   | Voltage (kV)                                     | `300`       |

### SLURM Settings
| Flag                          | Description                                      | Default     |
|-------------------------------|--------------------------------------------------|-------------|
| `--partition`                 | SLURM partition                                  | `emgpu`     |
| `--ntasks`                    | SLURM ntasks                                     | `1`         |
| `--nodes`                     | SLURM nodes                                      | `1`         |
| `--ntasks-per-node`           | SLURM tasks per node                             | `1`         |
| `--cpus-per-task`             | SLURM CPUs per task                              | `4`         |
| `--gres`                      | SLURM GPU resource (e.g. `gpu:1`)                | `gpu:1`     |
| `--mem`                       | SLURM memory (GB)                                | `128`       |
| `--qos`                       | SLURM Quality of Service                         | `emgpu`     |
| `--time`                      | SLURM time limit                                 | `05:00:00`  |
| `--mail-type`                 | SLURM mail notifications                         | `none`      |

---

## Example Output
After running, you’ll get one folder per tomogram under `submission/`:

```
submission/
├─ Position_1/
│  ├─ Position_1.tlt
│  ├─ Position_1_defocus.txt
│  ├─ Position_1_exposure.txt
│  └─ submit_Position_1.sh
├─ Position_2/
└─ Position_3/
```

Sample `submit_Position_1.sh`:

```bash
#!/bin/bash -l
#SBATCH -o pytom.out%j
#SBATCH -D ./
#SBATCH -J pytom_1
#SBATCH --partition=emgpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=none
#SBATCH --mem=128G
#SBATCH --qos=emgpu
#SBATCH --time=05:00:00

ml purge
ml pytom-match-pick

pytom_match_template.py \
  -v ../Position_1_Vol.mrc \
  -a submission/Position_1/Position_1.tlt \
  --dose-accumulation submission/Position_1/Position_1_exposure.txt \
  --defocus submission/Position_1/Position_1_defocus.txt \
  -t /path/to/template.mrc \
  -d submission/Position_1 \
  -m /path/to/mask.mrc \
  --angular-search 10 \
  -s 2 2 1 \
  --voxel-size-angstrom 7.64 \
  -r \
  --rng-seed 69 \
  -g 0 \
  --amplitude-contrast 0.07 \
  --spherical-aberration 2.7 \
  --voltage 300 \
  --per-tilt-weighting \
  --tomogram-ctf-model phase-flip \
  --non-spherical-mask
```

## Full Help Output
Run `./batch_pytom_aretomo3.py -h` to see all flags and defaults. Feel free to open an issue or submit a PR for further customization!

