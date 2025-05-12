#!/usr/bin/env python3
import os
import re
import argparse
import subprocess
from typing import Set

def parse_args():
    p = argparse.ArgumentParser(
        description='Batch-submit PyTom template matching for AreTomo3 tomograms')
    # I/O
    p.add_argument('-i', '--aretomo-dir',    required=True,
                   help='AreTomo3 output directory')
    p.add_argument('-d', '--output-dir',     default='submission',
                   help='Top-level folder for per-tomogram subfolders')
    p.add_argument('--bmask-dir',
               help='Directory containing binary mask files (format has to be: <prefix>.mrc !!)')
    # required PyTom inputs
    p.add_argument('-t', '--template',       required=True,
                   help='Template MRC file')
    p.add_argument('-m', '--mask',           required=True,
                   help='Mask MRC file')
    p.add_argument('-g', '--gpu-ids',        nargs='+', required=True,
                   help='GPU IDs for pytom_match_template.py (e.g. -g 0)')
    p.add_argument('--voxel-size-angstrom',  type=float, required=True,
                   help='Voxel size in Å')
    # dose & selection
    p.add_argument('--dose',                type=float, required=True,
                   help='Electron dose per tilt (e-/Å²)')
    p.add_argument('--include',             nargs='+',
                   help='Process only these prefixes')
    p.add_argument('--exclude',             nargs='+',
                   help='Exclude these prefixes')
    p.add_argument('--dry-run',             action='store_true',
                   help="Generate scripts only; don't sbatch")

    # require exactly one of particle-diameter or angular-search
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--particle-diameter', type=float,
                       help='Particle diameter in Å (Crowther sampling)')
    group.add_argument('--angular-search',
                       help='Override angular search')

    # PyTom optional flags
    p.add_argument('--non-spherical-mask',  action='store_true',
                   help='Enable non-spherical mask')
    p.add_argument('--z-axis-rotational-symmetry', type=int,
                   help='Z-axis rotational symmetry')
    p.add_argument('-s', '--volume-split',   nargs=3, type=int,
                   metavar=('X','Y','Z'),
                   help='Split volume into blocks X Y Z')
    p.add_argument('--search-x', nargs=2, type=int,
                   metavar=('START','END'),
                   help='Search range along x')
    p.add_argument('--search-y', nargs=2, type=int,
                   metavar=('START','END'),
                   help='Search range along y')
    p.add_argument('--search-z', nargs=2, type=int,
                   metavar=('START','END'),
                   help='Search range along z')
    p.add_argument('--tomogram-ctf-model',  choices=['phase-flip'],
                   help='CTF model used in reconstruction')
    p.add_argument('-r', '--random-phase-correction',
                   action='store_true',
                   help='Enable random-phase correction')
    p.add_argument('--half-precision',      action='store_true',
                   help='Use float16 output')
    p.add_argument('--rng-seed',            type=int, default=69,
                   help='RNG seed (default: %(default)s)')
    p.add_argument('--per-tilt-weighting',  action='store_true',
                   help='Enable per-tilt weighting')
    p.add_argument('--low-pass',            type=float,
                   help='Low-pass filter Å')
    p.add_argument('--high-pass',           type=float,
                   help='High-pass filter Å')

    # CTF & imaging defaults
    p.add_argument('--amplitude-contrast',  type=float, default=0.07,
                   help='Amplitude contrast (default: %(default)s)')
    p.add_argument('--spherical-aberration', type=float, default=2.7,
                   help='Spherical aberration mm (default: %(default)s)')
    p.add_argument('--voltage',             type=float, default=300,
                   help='Voltage kV (default: %(default)s)')
    p.add_argument('--phase-shift',         type=float,
                help='Phase shift degrees')
    p.add_argument('--defocus-handedness',  type=int, choices=[-1,0,1],
                   help='Defocus handedness (only if set)')
    p.add_argument('--spectral-whitening',  action='store_true',
                   help='Enable spectral whitening')
    p.add_argument('--relion5-tomograms-star',
                   help='RELION5 tomograms.star path (overrides tilts/CTF files)')
    p.add_argument('--log',                 choices=['info','debug'],
                   help='Logging level')

    # SLURM defaults
    p.add_argument('--partition',           default='emgpu',
                   help='SLURM partition')
    p.add_argument('--ntasks',              default='1',
                   help='SLURM ntasks')
    p.add_argument('--nodes',               default='1',
                   help='SLURM nodes')
    p.add_argument('--ntasks-per-node',     default='1',
                   help='SLURM tasks/node')
    p.add_argument('--cpus-per-task',       default='4',
                   help='SLURM CPUs/task')
    p.add_argument('--gres',                default='gpu:1',
                   help='SLURM gres')
    p.add_argument('--mem',                 default='128',
                   help='SLURM mem (GB)')
    p.add_argument('--qos',                 default='emgpu',
                   help='SLURM QoS')
    p.add_argument('--time',                default='05:00:00',
                   help='SLURM time')
    p.add_argument('--mail-type',           default='none',
                   help='SLURM mail-type')
    return p.parse_args()


def find_prefixes(aretomo_dir, include, exclude):
    all_mrc = [f for f in os.listdir(aretomo_dir)
               if f.endswith('.mrc')
                  and '_Vol' not in f
                  and '_EVN' not in f
                  and '_ODD' not in f
                  and '_CTF' not in f]
    prefixes = sorted(os.path.splitext(f)[0] for f in all_mrc)
    if include:
        prefixes = [p for p in prefixes
                    if any(re.match(f'^{pat.replace("*",".*")}$', p)
                           for pat in include)]
    if exclude:
        prefixes = [p for p in prefixes
                    if not any(re.match(f'^{pat.replace("*",".*")}$', p)
                               for pat in exclude)]
    return prefixes


def read_tlt_file(aretomo_dir, prefix):
    fn = os.path.join(aretomo_dir, f"{prefix}_Imod", f"{prefix}_st.tlt")
    with open(fn) as f:
        return [float(x) for x in f if x.strip()]


def read_tlt_file_from_aln(aretomo_dir, prefix):
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

def read_ctf_file(aretomo_dir, prefix):
    fn = os.path.join(aretomo_dir, f"{prefix}_CTF.txt")
    data = []
    with open(fn) as f:
        for L in f:
            if L.startswith('#') or not L.strip():
                continue
            parts = L.split()
            fr = int(parts[0])
            avg = 0.5*(float(parts[1]) + float(parts[2])) * 1e-4
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

def read_order_csv(aretomo_dir, prefix):
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


def calculate_cumulative_exposure(tilts, order, dose):
    expo = {}
    cum = 0.0
    for num, tilt in order:
        expo[round(tilt, 2)] = cum
        cum += dose
    return [expo[round(t, 2)] for t in tilts]


def write_aux_files(base_out, prefix, tilts, tilts_aln, ctf_filt, expos_filt):
    od = os.path.join(base_out, prefix)
    os.makedirs(od, exist_ok=True)
    tlt  = os.path.join(od, f"{prefix}.tlt")
    df   = os.path.join(od, f"{prefix}_defocus.txt")
    exp  = os.path.join(od, f"{prefix}_exposure.txt")
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


def make_sbatch(prefix, tlt, df, exp, args):
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
        f.write(f"#SBATCH --time={args.time}\n\n")
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


def submit(script, dry):
    if dry:
        print(f"[dry-run] sbatch {script}")
    else:
        subprocess.run(['sbatch', script], check=False)


def main():
    args = parse_args()
    args.aretomo_dir = os.path.abspath(args.aretomo_dir)
    args.output_dir  = os.path.abspath(args.output_dir)

    prefixes = find_prefixes(args.aretomo_dir, args.include, args.exclude)
    for p in prefixes:
        excl  = read_exclude(args.aretomo_dir, p)
        tilts = read_tlt_file(args.aretomo_dir, p)
        tilts_aln =read_tlt_file_from_aln(args.aretomo_dir, p)
        ctf   = read_ctf_file(args.aretomo_dir, p)
        ctf_filt = [d for d in ctf if d['frame'] not in excl]
        order = read_order_csv(args.aretomo_dir, p)
        expos = calculate_cumulative_exposure(tilts, order, args.dose)
        expos_filt = [expo for i, expo in enumerate(expos, start=1) if i not in excl]

        tlt, df, exp = write_aux_files(args.output_dir, p, tilts, tilts_aln, ctf_filt, expos_filt)
        sb_script    = make_sbatch(p, tlt, df, exp, args)
        submit(sb_script, args.dry_run)

    print("Done.")


if __name__ == '__main__':
    main()
