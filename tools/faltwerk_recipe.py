#!/usr/bin/env python3
"""Faltwerk standard recipe: ONE entry per source, no scene-specific parameters.

  faltwerk_recipe.py --source images|mesh --scene <scene.json> --out <run dir> [--rate-target MBIT] [--run]
                     [--reuse <stage>=<dir> ...] [--cfg tools/recipe_standard.json] [--py <python>]

Writes <out>/recipe.sh (the full chain, every command spelled out) and <out>/recipe.json (scene data paths, constants,
which rules ran where); --run executes recipe.sh (blocking; the GPU stages need CUDA + gsplat).

scene.json holds DATA only (paths, frame count, what exists), never a tuning value:
  {"name": "...", "gt": "<GT dir with cameras.json>", "frames": 64,
   "teacher": "<per-frame teacher PLY pattern with {:05d}>" (optional), "teacher_frames": 50 (optional),
   "kind": "llff" | "rig" (images: llff = cameras carry near/far bounds, N3DV-style; rig = matte'd multi-camera rig),
   mesh: "geom": [{"dir": "<geom dir>", "f0": 0, "frames": 444, "cams": "<cams.json>"}, ...], "gt_sil": .., "gt_shaded": ..,
         "looks": ["sil", "shaded"]}
The rate target is a DELIVERY value (e.g. 50 Mbit/s for a web/Quest player), not a scene value; without it nothing is
pruned (quality first).

Stages (images):
  llff : sparse depth (if missing) -> frame-0 init (p15_init, same switches for every scene) -> section training
         (n3dv_train / e7, near plane 0.9 x min camera near, static-lr 0.3) -> standard encode -> decoded eval + red/white
  rig + teacher: stage 1 tracking against the teacher (swap tau = 2.2 x median teacher sigma, extent of the visible init,
         random background + alpha loss because the GT has a matte) -> stage 2 curve fit (position tolerance = 1.0 x
         sigma_fine) -> fine-tune -> [rate target: contrib (every frame) -> budget from the rate target -> fine-tune the
         kept splats again] -> standard encode -> decoded eval + red/white
Stages (mesh): per geometry piece: mesh rules (h from the pixel footprint, per-mesh caps, backdrop split) -> mesh2gs
         (figures, backdrop) with the v2 splat fix -> standard encode per stream -> decoded streams.
Encode (all sources): train/recipe/encode.py (dead / giant / runaway cut, deep floor, cull floor, position step from
         the splat size, SH degree from the data).
"""
import argparse, json, os, shlex, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))


def q(x):
    return shlex.quote(str(x))


def env_str(d):
    return ' '.join(f'{k}={q(v)}' for k, v in d.items())


def images_llff(sc, cfg, out, reuse, code, root):
    ES = "".join(f" --set {q(s)}" for s in sc.get("enc_set", []))
    c = cfg['images']; F = int(sc['frames']); G = sc['gt']
    L = []
    L.append(f'[ -f {q(G)}/sparse_depth.npz ] || $PY -u train/llff/p15_prep.py sparse {q(G)} > {q(out)}/prep.log 2>&1 '
             f'|| {{ st "FAIL prep"; exit 1; }}')
    init = reuse.get('init', f'{out}/init')
    if 'init' not in reuse:
        L.append(f'[ -f {q(init)}/init.pt ] || {{ E3_GT={q(G)} $PY -u train/llff/p15_init.py --out {q(init)} --tag std '
                 f'{c["init"]} > {q(out)}/init.log 2>&1 && st "init ok $(grep -o \'P15 INIT OK.*\' {q(out)}/init.log | tail -1)" '
                 f'|| {{ st "FAIL init"; exit 1; }}; }}')
    tr = reuse.get('train', f'{out}/train')
    if 'train' not in reuse:
        L.append(f'mkdir -p {q(tr)}; [ -f {q(tr)}/init.pt ] || cp {q(init)}/init.pt {q(tr)}/init.pt; '
                 f'[ -f {q(tr)}/expo.pt ] || cp {q(init)}/expo.pt {q(tr)}/expo.pt')
        L.append(f'[ -f {q(tr)}/k5/curves.npz ] || {{ env E3_GT={q(G)} E7_RUN={q(tr)} P15_EXPO={q(tr)}/expo.pt {c["train_env"]} '
                 f'E7_FRAMES={F} $PY -u train/llff/n3dv_train.py > {q(out)}/train.log 2>&1 && st "train ok" '
                 f'|| {{ st "FAIL train"; exit 1; }}; }}')
    kd = f'{tr}/k5'
    if sc.get('rate_target'):
        L.append('st "rate target for llff not implemented (prune -> E7_FINETUNE); encoding without budget"')
    E = f'{out}/enc'
    L.append(f'N=falt; [ -f {q(E)}/csgs.json ] && N=csgs                 # stream dir: falt/ (runs before csgs/)')
    L.append(f'[ -f {q(E)}/$N.json ] || $PY -u train/recipe/encode.py {q(kd)} {q(E)} --source images --out-name $N --cams {q(G)}/cameras.json{ES} > {q(out)}/encode.log 2>&1 '
             f'&& st "$(grep P44ENC {q(out)}/encode.log)" || {{ st "FAIL encode"; exit 1; }}')
    L.append(f'[ -f {q(E)}/eval.json ] || E3_GT={q(G)} $PY -u train/llff/n3dv_eval.py {q(E)}/$N/ {q(E)}/eval.json --save-renders '
             f'> {q(out)}/eval.log 2>&1 || {{ st "FAIL eval"; exit 1; }}')
    L.append(f'grep -q mean_lpips_vgg {q(E)}/eval.json || E3_GT={q(G)} $PY -u train/llff/n3dv_lpips.py {q(E)}/eval.json --cuda '
             f'> {q(out)}/lpips.log 2>&1 || st "FAIL lpips"')
    L.append(f'grep -q \'"rw"\' {q(E)}/eval.json || E3_GT={q(G)} $PY -u train/llff/n3dv_rw.py {q(E)}/$N/ {q(E)}/eval.json --fstep 4 '
             f'> {q(out)}/rw.log 2>&1 || st "FAIL rw"')
    L.append(f'rm -rf {q(E)}/$N.renders; st "eval ok $(grep -o \'"mean_psnr\": [0-9.]*\' {q(E)}/eval.json)"')
    return L


def images_teacher(sc, cfg, out, reuse, code, root, rate_target):
    ES = "".join(f" --set {q(s)}" for s in sc.get("enc_set", []))
    c = cfg['images']; t = c['teacher']; F = int(sc['frames']); G = sc['gt']
    L = []
    ES += " --matte"                                             # rig GT with a matte: object scene -> size cuts on
    base = {'P8_FRAMES': F, 'P8_GT': G, 'P43_FRAMES': F, 'P43_TEACHER': sc['teacher'],
            'P43_PLYFRAMES': sc.get('teacher_frames', F)}
    base.update(c['matte'])                                      # the rig GT carries a matte -> random bg + alpha loss
    L.append(f'export {env_str(base)}')
    L.append('TS=$($PY train/recipe/teacher_stats.py "$P43_TEACHER" --minop 0.1) || exit 1')
    L.append('EXT=$(echo "$TS" | $PY -c "import json,sys; print(json.load(sys.stdin)[\'ext\'])"); '
             f'TAU=$(echo "$TS" | $PY -c "import json,sys; print({t["P43_SWAP_TAU_REL"]} * json.load(sys.stdin)[\'sigma_p50\'])")')
    L.append('st "teacher rules: ext $EXT (visible init), swap tau $TAU"')
    s1 = reuse.get('stage1', f'{out}/s1')
    if 'stage1' not in reuse:
        e1 = {k: v for k, v in t.items() if k.startswith(('P43_ITT', 'P43_LR', 'P43_REFRESH', 'E3_SH'))}
        L.append(f'[ -f {q(s1)}/s1.json ] || {{ mkdir -p {q(s1)}; env P43_RUN={q(s1)} P43_EXT=$EXT P43_SWAP_TAU=$TAU {env_str(e1)} '
                 f'$PY -u train/teacher/stage1.py > {q(s1)}/log 2>&1 && st "stage1 ok" || {{ st "FAIL stage1"; exit 1; }}; }}')
    fit = reuse.get('fit', f'{out}/fit')
    if 'fit' not in reuse:
        L.append(f'SF=$(echo "$TS" | $PY -c "import json,sys; print({t["TOLP_REL"]} * json.load(sys.stdin)[\'sigma_p10\'])")')
        L.append(f'[ -f {q(fit)}/fit.json ] || $PY -u train/teacher/stage2_fit.py {q(s1)} {q(fit)} --L {t["S2_L"]} --K {t["S2_K"]} '
                 f'--tols "P=$SF,R=20,S=0.3,O=0.05,D=0.08" > {q(out)}/fit.log 2>&1 || {{ st "FAIL fit"; exit 1; }}')
    K = int(t['S2_K'])
    kd = reuse.get('ft', f'{out}/k{K}')
    ft_env = {'P43_EXT': '$EXT', 'P43_MATTE': c['matte']['P43_MATTE'], 'P43_FT_LPIPS': t['P43_FT_LPIPS'], 'P43_FT_ITS': t['P43_FT_ITS']}
    fe = ' '.join(f'{k}={v}' for k, v in ft_env.items())
    if 'ft' not in reuse:
        L.append(f'[ -f {q(kd)}/curves.npz ] || {{ env {fe} $PY -u train/teacher/stage2_ft.py {q(fit)} {q(kd)} > {q(out)}/ft.log 2>&1 '
                 f'|| {{ st "FAIL ft"; exit 1; }}; }}')
    keep_arg = ''
    if rate_target:
        cd = reuse.get('contrib', f'{out}/contrib')
        if 'contrib' not in reuse:
            L.append(f'[ -f {q(cd)}/contrib.npz ] || $PY -u train/recipe/contrib.py {q(kd)} {q(cd)} --budgets 150000 '
                     f'--fstep {cfg["budget"]["contrib_fstep"]} > {q(out)}/contrib.log 2>&1 || {{ st "FAIL contrib"; exit 1; }}')
        L.append(f'[ -f {q(out)}/budget/keep.npz ] || $PY -u train/recipe/budget.py {q(cd)}/contrib.npz {q(kd)} {q(out)}/budget '
                 f'--target {rate_target} --source images --cams {q(G)}/cameras.json{ES} > {q(out)}/budget.log 2>&1 || {{ st "FAIL budget"; exit 1; }}')
        L.append(f'st "budget $(tail -1 {q(out)}/budget.log)"')
        kb = f'{out}/k{K}b'
        L.append(f'[ -f {q(kb)}/curves.npz ] || {{ env {fe} P43_FT_KEEP={q(out)}/budget/keep.npz $PY -u train/teacher/stage2_ft.py '
                 f'{q(kd)} {q(kb)} > {q(out)}/ft_keep.log 2>&1 && st "prune->ft ok" || {{ st "FAIL prune-ft"; exit 1; }}; }}')
        kd = kb; keep_arg = f' --keep {q(out)}/budget/keep.npz'
    E = f'{out}/enc'
    L.append(f'N=falt; [ -f {q(E)}/csgs.json ] && N=csgs                 # stream dir: falt/ (runs before csgs/)')
    L.append(f'[ -f {q(E)}/$N.json ] || $PY -u train/recipe/encode.py {q(kd)} {q(E)} --source images --out-name $N --cams {q(G)}/cameras.json{keep_arg}{ES} > {q(out)}/encode.log 2>&1 '
             f'&& st "$(grep P44ENC {q(out)}/encode.log)" || {{ st "FAIL encode"; exit 1; }}')
    L.append(f'[ -f {q(E)}/eval.json ] || $PY -u train/rig/rig_eval.py {q(E)}/$N/ {q(E)}/eval.json '
             f'> {q(out)}/eval.log 2>&1 || st "FAIL eval"')
    L.append(f'st "eval ok $(grep -o \'"mean_psnr_fg\": [0-9.]*\' {q(E)}/eval.json)"')
    return L


def mesh(sc, cfg, out, reuse, code, root):
    """rules per geometry piece pooled over the whole clip
    (one sample spacing h), mesh2gs per look and part (figures / backdrop), standard encode with the 0.5-px step at
    the figures' pixel footprint."""
    L = []
    pieces = sc['geom']
    for i, p in enumerate(pieces):                            # pass 1: per-piece mesh statistics
        L.append(f'[ -f {q(out)}/rules_p{i}_0.json ] || $PY train/recipe/mesh_std.py stats --three {q(p["dir"])} --frames {p["frames"]} '
                 f'--f0 {p.get("f0", 0)} --cams {q(p["cams"])} --json {q(out)}/rules_p{i}_0.json > /dev/null || {{ st "FAIL rules p{i}"; exit 1; }}')
    for i, p in enumerate(pieces):                            # pass 2: pooled over the clip
        pool = ','.join(f'{out}/rules_p{j}_0.json' for j in range(len(pieces)) if j != i)
        L.append(f'$PY train/recipe/mesh_std.py stats --three {q(p["dir"])} --frames {p["frames"]} --f0 {p.get("f0", 0)} '
                 f'--cams {q(p["cams"])} --pool {q(pool)} --json {q(out)}/rules_p{i}.json > {q(out)}/rules_p{i}_owners.json')
        for look in sc.get('looks', ['sil', 'shaded']):
            for part in ('fg', 'bd'):
                run = f'{out}/{look}_p{i}_{part}'
                L.append(f'ls {q(run)}/k*/mesh2gs.json > /dev/null 2>&1 || {{ grep -q "no meshes" {q(run)}.log 2>/dev/null || '
                         f'$PY -u train/recipe/mesh_std.py run --part {part} --three {q(p["dir"])} --frames {p["frames"]} '
                         f'--f0 {p.get("f0", 0)} --cams {q(p["cams"])} --pool {q(pool)} --out {q(run)} --look {look} > {q(run)}.log 2>&1 '
                         f'|| {{ st "FAIL mesh2gs {look} p{i} {part}"; exit 1; }}; }}')
                L.append(f'FP=$(grep -o \'"footprint_fig": [0-9.e-]*\' {q(out)}/rules_p{i}.json | head -1 | awk \'{{print $2}}\')')
                L.append(f'if ls {q(run)}/k*/mesh2gs.json > /dev/null 2>&1; then KD=$(ls -d {q(run)}/k*/ | head -1); '
                         f'N=falt; [ -f {q(run)}/enc/csgs.json ] && N=csgs; '
                         f'[ -f {q(run)}/enc/$N.json ] || $PY -u train/recipe/encode.py "${{KD%/}}" {q(run)}/enc --source mesh --out-name $N --footprint "$FP" '
                         f'> {q(run)}/encode.log 2>&1 && st "$(grep P44ENC {q(run)}/encode.log)" || {{ st "FAIL encode {look} p{i} {part}"; exit 1; }}; fi')
    L.append('st "mesh streams done: <out>/<look>_p<i>_<part>/enc/falt/"')
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=('images', 'mesh'), required=True)
    ap.add_argument('--scene', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--rate-target', type=float, default=None)
    ap.add_argument('--reuse', action='append', default=[], help='<stage>=<dir>: take a finished, IDENTICALLY configured stage')
    ap.add_argument('--cfg', default=os.path.join(REPO, 'tools', 'recipe_standard.json'))
    ap.add_argument('--py', default='python3', help='python of the training environment (torch, gsplat)')
    ap.add_argument('--code', default=None, help='code dir on the host (default: this checkout)')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--enc-set', action='append', default=[], help='A/B only: encoder constant override (k=v)')
    ap.add_argument('--write', default=None, help='local dir for recipe.sh/json when --out is a path on another (GPU) machine')
    a = ap.parse_args()
    sc = json.load(open(a.scene)); cfg = json.load(open(a.cfg))
    if a.rate_target:
        sc['rate_target'] = a.rate_target
    if a.enc_set:
        sc['enc_set'] = a.enc_set
    reuse = dict(r.split('=', 1) for r in a.reuse)
    code = a.code or REPO; root = None; out = a.out
    wdir = a.write or out
    os.makedirs(wdir, exist_ok=True)
    H = ['#!/usr/bin/env bash', f'# generated by tools/faltwerk_recipe.py --source {a.source} --scene {a.scene}',
         'set -uo pipefail', f'PY=${{PY:-{q(a.py)}}}; cd {q(code)} || exit 1', f'mkdir -p {q(out)}',
         f'export ROOT=${{ROOT:-{q(out)}}} PYTHONDONTWRITEBYTECODE=1',
         f'export P42_CODE=${{P42_CODE:-{q(code)}/train/mesh}}',
         f'st() {{ echo "$(date -Is) $*" | tee -a {q(out)}/status; }}', f'st "start {a.source} {sc.get("name", "")}"']
    if a.source == 'images':
        kind = sc.get('kind') or ('llff' if 'near' in json.dumps(json.load(open(os.path.join(sc['gt'], 'cameras.json')))['cameras'][0])
                                  else 'rig')
        if kind == 'llff':
            body = images_llff(sc, cfg, out, reuse, code, root)
        elif sc.get('teacher'):
            body = images_teacher(sc, cfg, out, reuse, code, root, a.rate_target)
        else:
            sys.exit('images/rig without teacher: use the P25/P40 trainer (not part of the standard yet)')
    else:
        body = mesh(sc, cfg, out, reuse, code, root)
    sh = '\n'.join(H + body + ['st done', ''])
    open(os.path.join(wdir, 'recipe.sh'), 'w').write(sh)
    json.dump({'source': a.source, 'scene': sc, 'reuse': reuse, 'cfg': cfg, 'rate_target': a.rate_target, 'out': out},
              open(os.path.join(wdir, 'recipe.json'), 'w'), indent=1)
    print(os.path.join(wdir, 'recipe.sh'))
    if a.run:
        sys.exit(subprocess.call(['bash', os.path.join(wdir, 'recipe.sh')]))


if __name__ == '__main__':
    main()
