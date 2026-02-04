#!/usr/bin/env python3
import os
import re
import argparse
import subprocess
from typing import Set, List, Optional


def parse_args():
    p = argparse.ArgumentParser(
        description='Batch-submit PyTom template matching for AreTomo3 tomograms'
    )
    # I/O
    p.add_argument('-i', '--aretomo-dir', required=True,
                   help='AreTomo3 output directory')
    p.add_argument('-d', '--output-dir', default='submission',
                   help='Top-level folder for per-tomogram subfolders')
    p.add_argument('--bmask-dir',
                   help='Directory containing binary mask files (format has to be: <prefix>.mrc !!)')

    # Submission mode
    p.add_argument('--mode', choices=['array', 'per-tomo'], default='array',
                   help='Submission mode: one SLURM array job (array) or one job per tomogram (per-tomo). Default: array')

    # required PyTom inputs
    p.add_argument('-t', '--template', required=True,
                   help='Template MRC file')
    p.add_argument('-m', '--mask', required=True,
                   help='Mask MRC file')
    p.add_argument('-g', '--gpu-ids', nargs='+', required=True,
                   help='GPU IDs for pytom_match_template.py (e.g. -g 0)')
    p.add_argument('--voxel-size-angstrom', type=float, required=True,
                   help='Voxel size in Å')
    p.add_argument('--dose', type=float, required=True,
                   help='Electron dose per tilt (e-/Å²)')
    p.add_argument('--include', nargs='+',
                   help='Process only these prefixes')
    p.add_argument('--exclude', nargs='+',
                   help='Exclude these prefixes')
    p.add_argument('--dry-run', action='store_true',
                   help="Generate scripts only; don't sbatch")

    # require exactly one of particle-diameter or angular-search
    group = p.add_mutually_exclusive_group(required=True)
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
    p.add_argument('--mem', default='128',
                   help='SLURM mem (GB)')
    p.add_argument('--qos', default='emgpu',
                   help='SLURM QoS')
    p.add_argument('--time', default='05:00:00',
                   help='SLURM time')
    p.add_argument('--mail-type', default='none',
                   help='SLURM mail-type')

    # Array controls
    p.add_argument('--array-max-parallel', type=int, default=8,
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
        prefixes = [p for p in prefixes
                    if any(re.match(f'^{pat.replace("*", ".*")}$', p)
                           for pat in include)]
    if exclude:
        prefixes = [p for p in prefixes
                    if not any(re.match(f'^{pat.replace("*", ".*")}$', p)
                               for pat in exclude)]
    return prefixes


def read_tlt_file(aretomo_dir: str, prefix: str):
    fn = os.path.join(aretomo_dir, f"{prefix}_Imod", f"{prefix}_st.tlt")
    with open(fn) as f:
        return [float(x) for x in f if x.strip()]


def read_tlt_file_from_aln(aretomo_dir: str, prefix: str):
    aln_fn = os.path.join(aretomo_dir, f"{prefix}.aln")
    tilts_aln = []
    with open(aln_fn) as fh:
        for line in fh:
            if line.startswith('#'):
                # once we hit the Local Alignment section, we're done
                if line.lstrip().startswith('# Local Alignment'):
                    break
                continue
            parts = line.split()
            # guard: ensure there's a 10th column
            if len(parts) >= 10:
                tilts_aln.append(float(parts[9]))
    return tilts_aln


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


def read_exclude(aretomo_dir: str, prefix: str) -> Set[int]:
    """Return set of excluded frame numbers from `tilt.com`."""
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


def calculate_cumulative_exposure(tilts, order, dose: float):
    expo = {}
    cum = 0.0
    for num, tilt in order:
        expo[round(tilt, 2)] = cum
        cum += dose
    return [expo[round(t, 2)] for t in tilts]


def write_aux_files(base_out: str, prefix: str, tilts, tilts_aln, ctf_filt, expos_filt):
    od = os.path.join(base_out, prefix)
    os.makedirs(od, exist_ok=True)
    tlt = os.path.join(od, f"{prefix}.tlt")
    df = os.path.join(od, f"{prefix}_defocus.txt")
    exp = os.path.join(od, f"{prefix}_exposure.txt")
    with open(tlt, 'w') as f:
        for t in tilts_aln:
            f.write(f"{t}\n")
    with open(df, 'w') as f:
        for d in ctf_filt:
            f.write(f"{d['defocus_um']}\n")
    with open(exp, 'w') as f:
        for e in expos_filt:
            f.write(f"{e}\n")
    return tlt, df, exp


def make_sbatch(prefix: str, tlt: str, df: str, exp: str, args):
    """Legacy: one sbatch script per tomogram."""
    od = os.path.join(args.output_dir, prefix)
    os.makedirs(od, exist_ok=True)
    script = os.path.join(od, f"submit_{prefix}.sh")
    tomo = os.path.join(args.aretomo_dir, f"{prefix}_Vol.mrc")

    # Check for matching mask file if bmask-dir is provided
    tomogram_mask = None
    if args.bmask_dir:
        potential_mask = os.path.join(args.bmask_dir, f"{prefix}.mrc")
        if os.path.exists(potential_mask):
            tomogram_mask = potential_mask

    with open(script, 'w') as f:
        f.write("#!/bin/bash -l\n")
        f.write("#SBATCH -o pytom.out%j\n")
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

        # Start pytom command
        f.write("pytom_match_template.py \\\n")
        # required args
        for flag, val in [
            ("-v", tomo),
            ("-a", tlt),
            ("--dose-accumulation", exp),
            ("--defocus", df),
            ("-t", args.template),
            ("-d", od),
            ("-m", args.mask),
        ]:
            f.write(f"  {flag} {val} \\\n")

        # particle/angle
        if args.particle_diameter:
            f.write(f"  --particle-diameter {args.particle_diameter} \\\n")
        if args.angular_search:
            f.write(f"  --angular-search {args.angular_search} \\\n")

        # additional optional flags
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
        if args.tomogram_ctf_model:
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
        # always include imaging defaults
        f.write(f"  --amplitude-contrast {args.amplitude_contrast} \\\n")
        f.write(f"  --spherical-aberration {args.spherical_aberration} \\\n")
        f.write(f"  --voltage {args.voltage} \\\n")

    os.chmod(script, 0o755)
    return script


def make_array_sbatch(prefixes: List[str], args):
    """
    Create a single SLURM array job script plus a prefixes list.
    Each array task reads one prefix, then runs pytom_match_template.py on that tomogram.
    """
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

    # Pre-render optional args that are constant across tasks
    opt_lines = []

    # particle/angle (mutually exclusive, but argparse already enforces)
    if args.particle_diameter is not None:
        opt_lines.append(f"  --particle-diameter {args.particle_diameter} \\")
    if args.angular_search is not None:
        opt_lines.append(f"  --angular-search {args.angular_search} \\")

    if args.z_axis_rotational_symmetry is not None:
        opt_lines.append(f"  --z-axis-rotational-symmetry {args.z_axis_rotational_symmetry} \\")
    if args.volume_split:
        x, y, z = args.volume_split
        opt_lines.append(f"  -s {x} {y} {z} \\")
    if args.search_x:
        opt_lines.append(f"  --search-x {args.search_x[0]} {args.search_x[1]} \\")
    if args.search_y:
        opt_lines.append(f"  --search-y {args.search_y[0]} {args.search_y[1]} \\")
    if args.search_z:
        opt_lines.append(f"  --search-z {args.search_z[0]} {args.search_z[1]} \\")
    if args.voxel_size_angstrom is not None:
        opt_lines.append(f"  --voxel-size-angstrom {args.voxel_size_angstrom} \\")
    if args.low_pass is not None:
        opt_lines.append(f"  --low-pass {args.low_pass} \\")
    if args.high_pass is not None:
        opt_lines.append(f"  --high-pass {args.high_pass} \\")
    if args.random_phase_correction:
        opt_lines.append("  -r \\")
        opt_lines.append(f"  --rng-seed {args.rng_seed} \\")
    opt_lines.append(f"  -g {' '.join(args.gpu_ids)} \\")
    if args.per_tilt_weighting:
        opt_lines.append("  --per-tilt-weighting \\")
    if args.tomogram_ctf_model is not None:
        opt_lines.append(f"  --tomogram-ctf-model {args.tomogram_ctf_model} \\")
    if args.non_spherical_mask:
        opt_lines.append("  --non-spherical-mask \\")
    if args.spectral_whitening:
        opt_lines.append("  --spectral-whitening \\")
    if args.half_precision:
        opt_lines.append("  --half-precision \\")
    if args.defocus_handedness is not None:
        opt_lines.append(f"  --defocus-handedness {args.defocus_handedness} \\")
    if args.phase_shift is not None:
        opt_lines.append(f"  --phase-shift {args.phase_shift} \\")
    if args.relion5_tomograms_star is not None:
        opt_lines.append(f"  --relion5-tomograms-star {args.relion5_tomograms_star} \\")
    if args.log is not None:
        opt_lines.append(f"  --log {args.log} \\")
    # imaging defaults
    opt_lines.append(f"  --amplitude-contrast {args.amplitude_contrast} \\")
    opt_lines.append(f"  --spherical-aberration {args.spherical_aberration} \\")
    opt_lines.append(f"  --voltage {args.voltage} \\")

    # bmask dir: handled per task in bash (if exists, pass --tomogram-mask)
    bmask_dir_line = ""
    if args.bmask_dir:
        bmask_dir_line = f'BMASK_DIR="{os.path.abspath(args.bmask_dir)}"'
    else:
        bmask_dir_line = 'BMASK_DIR=""'

    with open(script, "w") as f:
        f.write("#!/bin/bash -l\n")
        f.write("#SBATCH -D " + args.output_dir + "\n")
        f.write("#SBATCH -o pytom_%A_%a.out\n")
        f.write("##SBATCH -e pytom_%A_%a.err\n")
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
        f.write("#set -euo pipefail\n\n")
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

        f.write('TOMO_MASK_ARGS=""\n')
        f.write('if [[ -n "${BMASK_DIR}" && -f "${BMASK_DIR}/${PREFIX}.mrc" ]]; then\n')
        f.write('  TOMO_MASK_ARGS="--tomogram-mask ${BMASK_DIR}/${PREFIX}.mrc"\n')
        f.write('fi\n\n')

        f.write("pytom_match_template.py \\\n")
        f.write('  -v "${TOMO}" \\\n')
        f.write('  -a "${TLT}" \\\n')
        f.write('  --dose-accumulation "${EXP}" \\\n')
        f.write('  --defocus "${DF}" \\\n')
        f.write('  -t "${TEMPLATE}" \\\n')
        f.write('  -d "${OD}" \\\n')
        f.write('  -m "${PMASK}" \\\n')
        for line in opt_lines:
            f.write(line + "\n")
        f.write('  ${TOMO_MASK_ARGS}\n')

    os.chmod(script, 0o755)
    return script, prefix_list


def submit(script: str, dry: bool):
    if dry:
        print(f"[dry-run] sbatch {script}")
    else:
        subprocess.run(['sbatch', script], check=False)


def main():
    args = parse_args()
    args.aretomo_dir = os.path.abspath(args.aretomo_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    prefixes = find_prefixes(args.aretomo_dir, args.include, args.exclude)

    # Always generate aux files per tomogram (tlt/defocus/exposure) into output_dir/prefix/
    for pfx in prefixes:
        excl = read_exclude(args.aretomo_dir, pfx)
        tilts = read_tlt_file(args.aretomo_dir, pfx)
        tilts_aln = read_tlt_file_from_aln(args.aretomo_dir, pfx)
        ctf = read_ctf_file(args.aretomo_dir, pfx)
        ctf_filt = [d for d in ctf if d['frame'] not in excl]
        order = read_order_csv(args.aretomo_dir, pfx)
        expos = calculate_cumulative_exposure(tilts, order, args.dose)
        expos_filt = [expo for i, expo in enumerate(expos, start=1) if i not in excl]
        write_aux_files(args.output_dir, pfx, tilts, tilts_aln, ctf_filt, expos_filt)

    if args.mode == 'per-tomo':
        for pfx in prefixes:
            od = os.path.join(args.output_dir, pfx)
            tlt = os.path.join(od, f"{pfx}.tlt")
            df = os.path.join(od, f"{pfx}_defocus.txt")
            exp = os.path.join(od, f"{pfx}_exposure.txt")
            sb_script = make_sbatch(pfx, tlt, df, exp, args)
            submit(sb_script, args.dry_run)
    else:
        array_script, prefix_list = make_array_sbatch(prefixes, args)
        submit(array_script, args.dry_run)

    print("Done.")


if __name__ == '__main__':
    main()
