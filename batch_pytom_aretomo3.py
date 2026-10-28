#!/usr/bin/env python3
import os
import re
import argparse
import subprocess
import shlex
import fnmatch
from typing import Set, List, Optional


def parse_args():
    p = argparse.ArgumentParser(
        description='Batch-submit PyTom template matching for AreTomo3 tomograms'
    )
    # I/O
    p.add_argument('-i', '--aretomo-dir',
                   help='AreTomo3 output directory')
    p.add_argument('-d', '--output-dir', default='submission',
                   help='Top-level folder for per-tomogram subfolders')
    p.add_argument('--bmask-dir',
                   help='Directory containing binary mask files (format has to be: <prefix>.mrc !!)')

    # Workflow: extraction uses the *_job.json files created by template matching.
    p.add_argument('--workflow', choices=['match', 'extract'], default='match',
                   help='Run template matching (default) or submit candidate extraction as a SLURM array')
    p.add_argument('-n', '--n-candidates', type=int, default=1000,
                   help='Maximum candidates per tomogram in extract workflow (default: 1000)')
    p.add_argument('--relion5-compat', dest='relion5_compat', action='store_true', default=True,
                   help='Use RELION5 compatible extraction output (default: enabled)')
    p.add_argument('--no-relion5-compat', dest='relion5_compat', action='store_false',
                   help='Disable RELION5 compatible output in extraction')
    p.add_argument('--extract-args', default='', metavar='"OPTIONS"',
                   help='Additional arguments for pytom_extract_candidates.py (shell-style quoted string)')

    # Submission mode
    p.add_argument('--mode', choices=['array', 'per-tomo'], default='array',
                   help='Submission mode: one SLURM array job (array) or one job per tomogram (per-tomo). Default: array')

    # required PyTom inputs
    p.add_argument('-t', '--template',
                   help='Template MRC file')
    p.add_argument('-m', '--mask',
                   help='Mask MRC file')
    p.add_argument('-g', '--gpu-ids', nargs='+',
                   help='GPU IDs for pytom_match_template.py (e.g. -g 0)')
    p.add_argument('--voxel-size-angstrom', type=float,
                   help='Voxel size in Å')
    p.add_argument('--dose', type=float,
                   help='Electron dose per tilt (e-/Å²)')
    p.add_argument('--include', nargs='+',
                   help='Process only these prefixes')
    p.add_argument('--exclude', nargs='+',
                   help='Exclude these prefixes')
    p.add_argument('--dry-run', action='store_true',
                   help="Generate scripts only; don't sbatch")

    # require exactly one of particle-diameter or angular-search
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument('--particle-diameter', type=float,
                       help='Particle diameter in Å (Crowther sampling)')
    group.add_argument('--angular-search',
                       help='Override angular search')

    # PyTom optional flags
    p.add_argument('--non-spherical-mask', action='store_true',
                   help='Enable non-spherical mask')
    p.add_argument('--z-axis-rotational-symmetry', type=int,
                   help='Z-axis rotational symmetry')
    p.add_argument('-s', '--volume-split', nargs=3, type=int,
                   metavar=('X', 'Y', 'Z'),
                   help='Split volume into blocks X Y Z')
    p.add_argument('--search-x', nargs=2, type=int,
                   metavar=('START', 'END'),
                   help='Search range along x')
    p.add_argument('--search-y', nargs=2, type=int,
                   metavar=('START', 'END'),
                   help='Search range along y')
    p.add_argument('--search-z', nargs=2, type=int,
                   metavar=('START', 'END'),
                   help='Search range along z')
    p.add_argument('--tomogram-ctf-model', choices=['phase-flip'],
                   help='CTF model used in reconstruction')
    p.add_argument('-r', '--random-phase-correction',
                   action='store_true',
                   help='Enable random-phase correction')
    p.add_argument('--half-precision', action='store_true',
                   help='Use float16 output')
    p.add_argument('--rng-seed', type=int, default=69,
                   help='RNG seed (default: %(default)s)')
    p.add_argument('--per-tilt-weighting', action='store_true',
                   help='Enable per-tilt weighting')
    p.add_argument('--low-pass', type=float,
                   help='Low-pass filter Å')
    p.add_argument('--high-pass', type=float,
                   help='High-pass filter Å')

    # CTF & imaging defaults
    p.add_argument('--amplitude-contrast', type=float, default=0.07,
                   help='Amplitude contrast (default: %(default)s)')
    p.add_argument('--spherical-aberration', type=float, default=2.7,
                   help='Spherical aberration mm (default: %(default)s)')
    p.add_argument('--voltage', type=float, default=300,
                   help='Voltage kV (default: %(default)s)')
    p.add_argument('--phase-shift', type=float,
                   help='Phase shift degrees')
    p.add_argument('--defocus-handedness', type=int, choices=[-1, 0, 1],
                   help='Defocus handedness (only if set)')
    p.add_argument('--spectral-whitening', action='store_true',
                   help='Enable spectral whitening')
    p.add_argument('--relion5-tomograms-star',
                   help='RELION5 tomograms.star path (overrides tilts/CTF files)')
    p.add_argument('--log', choices=['info', 'debug'],
                   help='Logging level')

    # SLURM defaults
    p.add_argument('--partition', default='emgpu',
                   help='SLURM partition')
    p.add_argument('--ntasks', default='1',
                   help='SLURM ntasks')
    p.add_argument('--nodes', default='1',
                   help='SLURM nodes')
    p.add_argument('--ntasks-per-node', default='1',
                   help='SLURM tasks/node')
    p.add_argument('--cpus-per-task', default='4',
                   help='SLURM CPUs/task')
    p.add_argument('--gres', default='gpu:1',
                   help='SLURM gres')
    p.add_argument('--mem', default='120', type=int,
                   help='SLURM mem (GB)')
    p.add_argument('--qos', default='emgpu,emgpu-fast',
                   help='SLURM QoS')
    p.add_argument('--time', default='05:00:00',
                   help='SLURM time')
    p.add_argument('--mail-type', default='none',
                   help='SLURM mail-type')

    # Array controls
    p.add_argument('--array-max-parallel', '--max-parallel', type=int, default=20,
                   help='If set, submit as --array=0-N%%M to cap concurrent tasks to M (recommended on busy partitions).')

    # Node include/exclude
    p.add_argument('--exclude-nodes', nargs='+',
                   help='SLURM nodes to exclude (e.g. --exclude-nodes sgp02 sgp03)')
    p.add_argument('--include-nodes', nargs='+',
                   help='SLURM nodes to include via --nodelist (e.g. --include-nodes sgp02 sgp03)')

    return p.parse_args()


def find_prefixes(aretomo_dir: str, include: Optional[List[str]], exclude: Optional[List[str]]) -> List[str]:
    all_mrc = [f for f in os.listdir(aretomo_dir)
               if f.endswith('.mrc')
               and '_Vol' not in f
               and '_EVN' not in f
               and '_ODD' not in f
               and '_CTF' not in f]
    prefixes = sorted(os.path.splitext(f)[0] for f in all_mrc)

    if include:
        include = include[0].split(',') if len(include) == 1 else include
        prefixes = [p for p in prefixes
                    if any(re.match(f'^{pat.replace("*", ".*")}$', p)
                           for pat in include)]
    if exclude:
        exclude = exclude[0].split(',') if len(exclude) == 1 else exclude
        prefixes = [p for p in prefixes
                    if not any(re.match(f'^{pat.replace("*", ".*")}$', p)
                               for pat in exclude)]
    return prefixes


def read_tlt_file(aretomo_dir: str, prefix: str):
    fn = os.path.join(aretomo_dir, f"{prefix}_Imod", f"{prefix}_st.tlt")
    if not os.path.isfile(fn):
        fn = os.path.join(aretomo_dir, f"{prefix}_Imod", f"{prefix}.tlt")
        if not os.path.isfile(fn):
            raise FileNotFoundError(f"No tilt file found for prefix {prefix} in {os.path.join(aretomo_dir, f'{prefix}_Imod')}")
    with open(fn) as f:
        return [float(x) for x in f if x.strip()]


def read_tlt_file_from_aln(aretomo_dir: str, prefix: str):
    aln_fn = os.path.join(aretomo_dir, f"{prefix}.aln")
    tilts_aln = []
    if not os.path.isfile(aln_fn):
        return tilts_aln
    with open(aln_fn) as fh:
        for line in fh:
            if line.startswith('#'):
                if line.lstrip().startswith('# Local Alignment'):
                    break
                continue
            parts = line.split()
            if len(parts) >= 10:
                tilts_aln.append(float(parts[9]))
    return tilts_aln


def read_darkframes_from_aln(aretomo_dir: str, prefix: str) -> Set[int]:
    # Read AreTomo DarkFrame exclusions from the global alignment header.
    aln_fn = os.path.join(aretomo_dir, f"{prefix}.aln")
    excl = set()
    if not os.path.isfile(aln_fn):
        return excl

    with open(aln_fn) as fh:
        for line in fh:
            s = line.lstrip()
            if s.startswith('# Local Alignment'):
                break
            if s.startswith('# DarkFrame'):
                m = re.search(r'DarkFrame\s*=\s*(\d+)\s+(\d+)\s+([-\d.]+)', s)
                if m:
                    excl.add(int(m.group(2)))
    return excl


def read_ctf_file(aretomo_dir: str, prefix: str):
    fn = os.path.join(aretomo_dir, f"{prefix}_CTF.txt")
    data = []
    with open(fn) as f:
        for L in f:
            if L.startswith('#') or not L.strip():
                continue
            parts = L.split()
            fr = int(parts[0])
            avg = 0.5 * (float(parts[1]) + float(parts[2])) * 1e-4
            data.append({'frame': fr, 'defocus_um': avg})
    return data


def read_ctf_file_imod_from_path(fn: str):
    data = []
    with open(fn) as f:
        for L in f:
            parts = L.split()
            if len(parts) == 7:
                fr = int(parts[0])
                avg = 0.5 * (float(parts[4]) + float(parts[5])) * 1e-3
                data.append({'frame': fr, 'defocus_um': avg})
    return data


def read_ctf_file_any(aretomo_dir: str, prefix: str):
    # Prefer standard AreTomo top-level CTF outputs, then fall back to _Imod variants.
    top_ctf = os.path.join(aretomo_dir, f"{prefix}_CTF.txt")
    if os.path.isfile(top_ctf):
        return read_ctf_file(aretomo_dir, prefix)

    top_ctf_imod = os.path.join(aretomo_dir, f"{prefix}_CTF_Imod.txt")
    if os.path.isfile(top_ctf_imod):
        return read_ctf_file_imod_from_path(top_ctf_imod)

    imod_candidates = [
        os.path.join(aretomo_dir, f"{prefix}_Imod", f"{prefix}_st_ctf.txt"),
        os.path.join(aretomo_dir, f"{prefix}_Imod", f"{prefix}_ctf.txt"),
    ]
    for fn in imod_candidates:
        if os.path.isfile(fn):
            return read_ctf_file_imod_from_path(fn)

    return None


def read_exclude(aretomo_dir: str, prefix: str) -> Set[int]:
    path = os.path.join(aretomo_dir, f"{prefix}_Imod", "tilt.com")
    if not os.path.isfile(path):
        return set()
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith("EXCLUDELIST"):
                return {int(x) for x in re.findall(r"\d+", line)}
    return set()


def read_order_csv(aretomo_dir: str, prefix: str):
    fn = os.path.join(aretomo_dir, f"{prefix}_Imod", f"{prefix}_order_list.csv")
    order = []
    with open(fn) as f:
        hdr = f.readline()
        if 'ImageNumber' not in hdr:
            f.seek(0)
        for L in f:
            if not L.strip():
                continue
            n, a = L.split(',')[:2]
            order.append((int(n), float(a)))
    return order


def calculate_cumulative_exposure_by_tilt(tilts_for_output, order, dose: float):
    # Assign cumulative dose in acquisition order, then write it back in final tilt-list order.
    expo_by_tilt = {}
    cum = 0.0
    for _frame_num, tilt in order:
        expo_by_tilt[round(float(tilt), 2)] = cum
        cum += dose

    missing_expo = [t for t in tilts_for_output if round(float(t), 2) not in expo_by_tilt]
    if missing_expo:
        raise RuntimeError(
            f"Missing exposure values for output tilts: {missing_expo}"
        )

    return [expo_by_tilt[round(float(t), 2)] for t in tilts_for_output]


def write_aux_files(base_out: str, prefix: str, tlt_values, defocus_values, exposure_values):
    od = os.path.join(base_out, prefix)
    os.makedirs(od, exist_ok=True)
    tlt = os.path.join(od, f"{prefix}.tlt")
    df = os.path.join(od, f"{prefix}_defocus.txt")
    exp = os.path.join(od, f"{prefix}_exposure.txt")

    with open(tlt, 'w') as f:
        for t in tlt_values:
            f.write(f"{t}\n")

    if defocus_values is not None:
        with open(df, 'w') as f:
            for d in defocus_values:
                f.write(f"{d}\n")
    else:
        if os.path.exists(df):
            os.remove(df)
        df = None

    with open(exp, 'w') as f:
        for e in exposure_values:
            f.write(f"{e}\n")

    return tlt, df, exp


def make_sbatch(prefix: str, tlt: str, df: Optional[str], exp: str, args):
    od = os.path.join(args.output_dir, prefix)
    os.makedirs(od, exist_ok=True)
    script = os.path.join(od, f"submit_{prefix}.sh")
    tomo = os.path.join(args.aretomo_dir, f"{prefix}_Vol.mrc")

    tomogram_mask = None
    if args.bmask_dir:
        potential_mask = os.path.join(args.bmask_dir, f"{prefix}.mrc")
        if os.path.exists(potential_mask):
            tomogram_mask = potential_mask

    with open(script, 'w') as f:
        f.write("#!/bin/bash -l\n")
        f.write("#SBATCH -o logs/pytom.out%j\n")
        f.write("#SBATCH -D ./\n")
        f.write(f"#SBATCH -J pytom_{prefix.split('_')[-1]}\n")
        f.write(f"#SBATCH --partition={args.partition}\n")
        f.write(f"#SBATCH --ntasks={args.ntasks}\n")
        f.write(f"#SBATCH --nodes={args.nodes}\n")
        f.write(f"#SBATCH --ntasks-per-node={args.ntasks_per_node}\n")
        f.write(f"#SBATCH --cpus-per-task={args.cpus_per_task}\n")
        f.write(f"#SBATCH --gres={args.gres}\n")
        f.write(f"#SBATCH --mail-type={args.mail_type}\n")
        f.write(f"#SBATCH --mem={args.mem}G\n")
        f.write(f"#SBATCH --qos={args.qos}\n")
        f.write(f"#SBATCH --time={args.time}\n")
        if args.exclude_nodes:
            f.write(f"#SBATCH --exclude={','.join(args.exclude_nodes)}\n")
        if args.include_nodes:
            f.write(f"#SBATCH --nodelist={','.join(args.include_nodes)}\n")
        f.write("\n")
        f.write("ml purge\nml pytom-match-pick\n\n")

        f.write("pytom_match_template.py \\\n")

        required_args = [
            ("-v", tomo),
            ("-a", tlt),
            ("--dose-accumulation", exp),
            ("-t", args.template),
            ("-d", od),
            ("-m", args.mask),
        ]
        if df is not None:
            required_args.insert(3, ("--defocus", df))

        for flag, val in required_args:
            f.write(f"  {flag} {val} \\\n")

        if args.particle_diameter:
            f.write(f"  --particle-diameter {args.particle_diameter} \\\n")
        if args.angular_search:
            f.write(f"  --angular-search {args.angular_search} \\\n")

        if args.z_axis_rotational_symmetry:
            f.write(f"  --z-axis-rotational-symmetry {args.z_axis_rotational_symmetry} \\\n")
        if args.volume_split:
            x, y, z = args.volume_split
            f.write(f"  -s {x} {y} {z} \\\n")
        if args.search_x:
            f.write(f"  --search-x {args.search_x[0]} {args.search_x[1]} \\\n")
        if args.search_y:
            f.write(f"  --search-y {args.search_y[0]} {args.search_y[1]} \\\n")
        if args.search_z:
            f.write(f"  --search-z {args.search_z[0]} {args.search_z[1]} \\\n")
        if args.voxel_size_angstrom:
            f.write(f"  --voxel-size-angstrom {args.voxel_size_angstrom} \\\n")
        if args.low_pass:
            f.write(f"  --low-pass {args.low_pass} \\\n")
        if args.high_pass:
            f.write(f"  --high-pass {args.high_pass} \\\n")
        if args.random_phase_correction:
            f.write("  -r \\\n")
            f.write(f"  --rng-seed {args.rng_seed} \\\n")
        f.write(f"  -g {' '.join(args.gpu_ids)} \\\n")
        if args.per_tilt_weighting:
            f.write("  --per-tilt-weighting \\\n")
        if args.tomogram_ctf_model and df is not None:
            f.write(f"  --tomogram-ctf-model {args.tomogram_ctf_model} \\\n")
        if args.non_spherical_mask:
            f.write("  --non-spherical-mask \\\n")
        if args.spectral_whitening:
            f.write("  --spectral-whitening \\\n")
        if args.half_precision:
            f.write("  --half-precision \\\n")
        if args.defocus_handedness is not None:
            f.write(f"  --defocus-handedness {args.defocus_handedness} \\\n")
        if tomogram_mask:
            f.write(f"  --tomogram-mask {tomogram_mask} \\\n")
        if args.phase_shift is not None:
            f.write(f"  --phase-shift {args.phase_shift} \\\n")
        if args.relion5_tomograms_star:
            f.write(f"  --relion5-tomograms-star {args.relion5_tomograms_star} \\\n")
        if args.log:
            f.write(f"  --log {args.log} \\\n")
        f.write(f"  --amplitude-contrast {args.amplitude_contrast} \\\n")
        f.write(f"  --spherical-aberration {args.spherical_aberration} \\\n")
        f.write(f"  --voltage {args.voltage} \n")

    os.chmod(script, 0o755)
    return script


def make_array_sbatch(prefixes: List[str], args):
    os.makedirs(args.output_dir, exist_ok=True)

    prefix_list = os.path.join(args.output_dir, "prefixes.txt")
    with open(prefix_list, "w") as fh:
        for p in prefixes:
            fh.write(f"{p}\n")

    n = len(prefixes)
    if n == 0:
        raise RuntimeError("No tomograms found after include/exclude filtering.")

    array_range = f"0-{n-1}"
    if args.array_max_parallel is not None:
        array_range = f"{array_range}%{args.array_max_parallel}"

    script = os.path.join(args.output_dir, "submit_array.sh")

    opt_lines = []

    if args.particle_diameter is not None:
        opt_lines.append(f'CMD+=(--particle-diameter {args.particle_diameter})')
    if args.angular_search is not None:
        opt_lines.append(f'CMD+=(--angular-search {args.angular_search})')

    if args.z_axis_rotational_symmetry is not None:
        opt_lines.append(f'CMD+=(--z-axis-rotational-symmetry {args.z_axis_rotational_symmetry})')
    if args.volume_split:
        x, y, z = args.volume_split
        opt_lines.append(f'CMD+=(-s {x} {y} {z})')
    if args.search_x:
        opt_lines.append(f'CMD+=(--search-x {args.search_x[0]} {args.search_x[1]})')
    if args.search_y:
        opt_lines.append(f'CMD+=(--search-y {args.search_y[0]} {args.search_y[1]})')
    if args.search_z:
        opt_lines.append(f'CMD+=(--search-z {args.search_z[0]} {args.search_z[1]})')
    if args.voxel_size_angstrom is not None:
        opt_lines.append(f'CMD+=(--voxel-size-angstrom {args.voxel_size_angstrom})')
    if args.low_pass is not None:
        opt_lines.append(f'CMD+=(--low-pass {args.low_pass})')
    if args.high_pass is not None:
        opt_lines.append(f'CMD+=(--high-pass {args.high_pass})')
    if args.random_phase_correction:
        opt_lines.append('CMD+=(-r)')
        opt_lines.append(f'CMD+=(--rng-seed {args.rng_seed})')
    opt_lines.append(f"CMD+=(-g {' '.join(args.gpu_ids)})")
    if args.per_tilt_weighting:
        opt_lines.append('CMD+=(--per-tilt-weighting)')
    if args.non_spherical_mask:
        opt_lines.append('CMD+=(--non-spherical-mask)')
    if args.spectral_whitening:
        opt_lines.append('CMD+=(--spectral-whitening)')
    if args.half_precision:
        opt_lines.append('CMD+=(--half-precision)')
    if args.defocus_handedness is not None:
        opt_lines.append(f'CMD+=(--defocus-handedness {args.defocus_handedness})')
    if args.phase_shift is not None:
        opt_lines.append(f'CMD+=(--phase-shift {args.phase_shift})')
    if args.relion5_tomograms_star is not None:
        opt_lines.append(f'CMD+=(--relion5-tomograms-star {args.relion5_tomograms_star})')
    if args.log is not None:
        opt_lines.append(f'CMD+=(--log {args.log})')
    opt_lines.append(f'CMD+=(--amplitude-contrast {args.amplitude_contrast})')
    opt_lines.append(f'CMD+=(--spherical-aberration {args.spherical_aberration})')
    opt_lines.append(f'CMD+=(--voltage {args.voltage})')

    if args.bmask_dir:
        bmask_dir_line = f'BMASK_DIR="{os.path.abspath(args.bmask_dir)}"'
    else:
        bmask_dir_line = 'BMASK_DIR=""'

    with open(script, "w") as f:
        f.write("#!/bin/bash -l\n")
        f.write("#SBATCH -D " + args.output_dir + "\n")
        f.write("#SBATCH -o logs/pytom_%A_%a.out\n")
        # f.write("##SBATCH -e logs/pytom_%A_%a.err\n")
        f.write(f"#SBATCH -J pytom_tm\n")
        f.write(f"#SBATCH --partition={args.partition}\n")
        f.write(f"#SBATCH --array={array_range}\n")
        f.write(f"#SBATCH --ntasks={args.ntasks}\n")
        f.write(f"#SBATCH --nodes={args.nodes}\n")
        f.write(f"#SBATCH --ntasks-per-node={args.ntasks_per_node}\n")
        f.write(f"#SBATCH --cpus-per-task={args.cpus_per_task}\n")
        f.write(f"#SBATCH --gres={args.gres}\n")
        f.write(f"#SBATCH --mail-type={args.mail_type}\n")
        f.write(f"#SBATCH --mem={args.mem}G\n")
        f.write(f"#SBATCH --qos={args.qos}\n")
        f.write(f"#SBATCH --time={args.time}\n")
        if args.exclude_nodes:
            f.write(f"#SBATCH --exclude={','.join(args.exclude_nodes)}\n")
        if args.include_nodes:
            f.write(f"#SBATCH --nodelist={','.join(args.include_nodes)}\n")
        f.write("\n")
        f.write("ml purge\nml pytom-match-pick\n\n")

        f.write(f'PREFIX_LIST="{prefix_list}"\n')
        f.write('PREFIX=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${PREFIX_LIST}")\n')
        f.write('if [[ -z "${PREFIX}" ]]; then echo "Empty prefix for task ${SLURM_ARRAY_TASK_ID}"; exit 2; fi\n\n')

        f.write(f'ARETOMO_DIR="{args.aretomo_dir}"\n')
        f.write(f'OUT_DIR="{args.output_dir}"\n')
        f.write(bmask_dir_line + "\n")
        f.write(f'TEMPLATE="{os.path.abspath(args.template)}"\n')
        f.write(f'PMASK="{os.path.abspath(args.mask)}"\n\n')

        f.write('OD="${OUT_DIR}/${PREFIX}"\n')
        f.write('mkdir -p "${OD}"\n')
        f.write('TOMO="${ARETOMO_DIR}/${PREFIX}_Vol.mrc"\n')
        f.write('TLT="${OD}/${PREFIX}.tlt"\n')
        f.write('DF="${OD}/${PREFIX}_defocus.txt"\n')
        f.write('EXP="${OD}/${PREFIX}_exposure.txt"\n\n')

        f.write('# Build command dynamically so CTF-dependent arguments are only passed when present.\n')
        f.write('CMD=(pytom_match_template.py)\n')
        f.write('CMD+=(-v "${TOMO}")\n')
        f.write('CMD+=(-a "${TLT}")\n')
        f.write('CMD+=(--dose-accumulation "${EXP}")\n')
        f.write('CMD+=(-t "${TEMPLATE}")\n')
        f.write('CMD+=(-d "${OD}")\n')
        f.write('CMD+=(-m "${PMASK}")\n\n')

        f.write('if [[ -s "${DF}" ]]; then\n')
        f.write('  CMD+=(--defocus "${DF}")\n')
        if args.tomogram_ctf_model is not None:
            f.write(f'  CMD+=(--tomogram-ctf-model {args.tomogram_ctf_model})\n')
        f.write('fi\n\n')

        f.write('if [[ -n "${BMASK_DIR}" && -f "${BMASK_DIR}/${PREFIX}.mrc" ]]; then\n')
        f.write('  CMD+=(--tomogram-mask "${BMASK_DIR}/${PREFIX}.mrc")\n')
        f.write('fi\n\n')

        for line in opt_lines:
            f.write(line + "\n")
        f.write('\n"${CMD[@]}"\n')

    os.chmod(script, 0o755)
    return script, prefix_list


def make_extract_array_sbatch(args):
    """One job JSON per array element, from previous PyTom template matching."""
    root = os.path.abspath(args.output_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Matching output directory does not exist: {root}")
    jobs = sorted(
        str(p.resolve()) for p in __import__('pathlib').Path(root).rglob('*_job.json')
        if p.is_file()
    )
    def patterns(values):
        return ([x for value in values for x in value.split(',')] if values else [])
    include, exclude = patterns(args.include), patterns(args.exclude)
    def selected(path):
        parent = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path).removesuffix('_job.json')
        return (not include or any(fnmatch.fnmatch(v, pat) for v in (parent, name) for pat in include)) and not any(
            fnmatch.fnmatch(v, pat) for v in (parent, name) for pat in exclude)
    jobs = [path for path in jobs if selected(path)]
    if not jobs:
        raise RuntimeError(f"No *_job.json files found under {root} after filtering")
    extraction_masks = []
    if args.bmask_dir:
        masks_dir = __import__('pathlib').Path(args.bmask_dir).resolve()
        if not masks_dir.is_dir():
            raise NotADirectoryError(f'Extraction mask directory not found: {masks_dir}')
        for job in jobs:
            job_path = __import__('pathlib').Path(job)
            names = [job_path.parent.name, job_path.name.removesuffix('_job.json')]
            matches = [masks_dir / f'{name}.mrc' for name in dict.fromkeys(names)
                       if (masks_dir / f'{name}.mrc').is_file()]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f'Expected exactly one extraction mask for {job}; tried {names} in {masks_dir} '
                    f'(found {len(matches)}).'
                )
            extraction_masks.append(matches[0])
    if args.particle_diameter is None:
        diameter = 300.0
    else:
        diameter = args.particle_diameter
    if diameter <= 0 or args.n_candidates <= 0 or args.array_max_parallel is not None and args.array_max_parallel <= 0:
        raise ValueError('Particle diameter, candidate count, and concurrency must be positive')
    extras = shlex.split(args.extract_args)
    blocked = {'-j', '--job', '--particle-diameter', '-n', '--n-candidates', '--relion5-compat', '--tomogram-mask'}
    if any(x in blocked or x.startswith('--particle-diameter=') for x in extras):
        raise ValueError('Do not override managed extraction flags in --extract-args')
    manifest = os.path.join(root, 'extract_jobs.txt')
    with open(manifest, 'w') as f:
        f.write('\n'.join(jobs) + '\n')
    if args.bmask_dir:
        with open(os.path.join(root, 'extract_masks.txt'), 'w') as f:
            f.write('\n'.join(map(str, extraction_masks)) + '\n')
    limit = f'%{args.array_max_parallel}' if args.array_max_parallel is not None else ''
    script = os.path.join(root, 'submit_extract_array.sh')
    lines = [
        '#!/bin/bash -l',
        f'#SBATCH -D {root}',
        '#SBATCH -o {root}/logs/pytom_extract_%A_%a.out',
        '#SBATCH -J pytom_extract',
        f'#SBATCH --partition={args.partition}',
        f'#SBATCH --array=0-{len(jobs)-1}{limit}',
        f'#SBATCH --ntasks={args.ntasks}',
        f'#SBATCH --nodes={args.nodes}',
        f'#SBATCH --ntasks-per-node={args.ntasks_per_node}',
        f'#SBATCH --cpus-per-task={args.cpus_per_task}',
        '#SBATCH --gres=gpu:1',  # Extraction supports one GPU only.
        f'#SBATCH --mail-type={args.mail_type}',
        f'#SBATCH --mem={args.mem}G',
        f'#SBATCH --qos={args.qos}',
        f'#SBATCH --time={args.time}',
    ]
    if args.exclude_nodes:
        lines.append(f'#SBATCH --exclude={",".join(args.exclude_nodes)}')
    if args.include_nodes:
        lines.append(f'#SBATCH --nodelist={",".join(args.include_nodes)}')
    lines.extend([
        '', 'ml purge', 'ml pytom-match-pick', '',
        f'MANIFEST={shlex.quote(manifest)}',
        *( [f"MASK_MANIFEST={shlex.quote(os.path.join(root, 'extract_masks.txt'))}"] if args.bmask_dir else [] ),
        'JOB=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")',
        '[[ -n "$JOB" && -s "$JOB" ]] || { echo "Missing job JSON: $JOB" >&2; exit 2; }',
        'CMD=(pytom_extract_candidates.py -j "$JOB")',
        *( ['MASK=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MASK_MANIFEST")',
            '[[ -n "$MASK" && -f "$MASK" ]] || { echo "Missing extraction mask: $MASK" >&2; exit 2; }']
           if args.bmask_dir else [] ),
        f'CMD+=(--particle-diameter {diameter:g})',
        f'CMD+=(-n {args.n_candidates})',
    ])
    if args.bmask_dir:
        lines.append('CMD+=(--tomogram-mask "$MASK")')
    if args.relion5_compat:
        lines.append('CMD+=(--relion5-compat)')
    if extras:
        lines.append('CMD+=(' + ' '.join(shlex.quote(x) for x in extras) + ')')
    lines.extend(['printf "Command: "; printf "%q " "${CMD[@]}"; echo', '"${CMD[@]}"', ''])
    with open(script, 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(script, 0o755)
    print(f'Extraction: {len(jobs)} job JSON files; 1 GPU per task')
    print(f'Manifest: {manifest}\nSBATCH: {script}')
    return script


def submit(script: str, dry: bool):
    if dry:
        print(f"[dry-run] sbatch {script}")
    else:
        subprocess.run(['sbatch', script], check=False)


def main():
    args = parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    if args.workflow == 'extract':
        if args.gres != 'gpu:1':
            print('NOTE: extraction always uses --gres=gpu:1 (ignoring --gres).')
        script = make_extract_array_sbatch(args)
        submit(script, args.dry_run)
        print('Done.')
        return
    # Preserve template matching requirements, but only in match workflow.
    required = {'--aretomo-dir': args.aretomo_dir, '--template': args.template,
                '--mask': args.mask, '--gpu-ids': args.gpu_ids,
                '--voxel-size-angstrom': args.voxel_size_angstrom, '--dose': args.dose}
    missing = [name for name, value in required.items() if value is None]
    if missing or (args.particle_diameter is None and args.angular_search is None):
        raise SystemExit('Template matching requires ' + ', '.join(missing) +
                         (' and --particle-diameter or --angular-search' if args.particle_diameter is None and args.angular_search is None else ''))
    args.aretomo_dir = os.path.abspath(args.aretomo_dir)
    prefixes = find_prefixes(args.aretomo_dir, args.include, args.exclude)

    for pfx in prefixes:
        raw_tilts = read_tlt_file(args.aretomo_dir, pfx)
        tilts_aln = read_tlt_file_from_aln(args.aretomo_dir, pfx)

        # Combine IMOD EXCLUDELIST and AreTomo DarkFrame exclusions on the same 1-based frame index.
        excl_imod = read_exclude(args.aretomo_dir, pfx)
        excl_dark = read_darkframes_from_aln(args.aretomo_dir, pfx)
        excl_all = excl_imod | excl_dark

        kept_frames = [i for i in range(1, len(raw_tilts) + 1) if i not in excl_all]

        # Prefer the aligned .aln tilt list when available because it matches the reconstructed volume state.
        if tilts_aln:
            tlt_out = tilts_aln
            if len(tlt_out) != len(kept_frames):
                raise RuntimeError(
                    f"{pfx}: .aln tilt count ({len(tlt_out)}) does not match kept frame count ({len(kept_frames)})."
                )
        else:
            tlt_out = [raw_tilts[i - 1] for i in kept_frames]

        ctf = read_ctf_file_any(args.aretomo_dir, pfx)
        if ctf is None:
            if args.tomogram_ctf_model:
                raise FileNotFoundError(
                    f"{pfx}: no CTF file found, but --tomogram-ctf-model was requested."
                )
            defocus_out = None
        else:
            ctf_by_frame = {d['frame']: d['defocus_um'] for d in ctf}
            missing_ctf = [i for i in kept_frames if i not in ctf_by_frame]
            if missing_ctf:
                raise RuntimeError(
                    f"{pfx}: missing CTF values for kept frames: {missing_ctf}"
                )
            defocus_out = [ctf_by_frame[i] for i in kept_frames]

        order = read_order_csv(args.aretomo_dir, pfx)
        exposure_out = calculate_cumulative_exposure_by_tilt(tlt_out, order, args.dose)

        write_aux_files(args.output_dir, pfx, tlt_out, defocus_out, exposure_out)

    if args.mode == 'per-tomo':
        for pfx in prefixes:
            od = os.path.join(args.output_dir, pfx)
            tlt = os.path.join(od, f"{pfx}.tlt")
            df = os.path.join(od, f"{pfx}_defocus.txt")
            exp = os.path.join(od, f"{pfx}_exposure.txt")
            if not os.path.isfile(df) or os.path.getsize(df) == 0:
                df = None
            sb_script = make_sbatch(pfx, tlt, df, exp, args)
            submit(sb_script, args.dry_run)
    else:
        array_script, prefix_list = make_array_sbatch(prefixes, args)
        submit(array_script, args.dry_run)

    print("Done.")


if __name__ == '__main__':
    main()
