#!/usr/bin/env python3
"""E7 – motion curves trained jointly per section (CurveStream-GS P1; GPU, train venv, gsplat).

Model: a pool of splats with lifetimes [birth, death) (frames). Per section of L frames (default 32 = codec GOP) every
time-dependent attribute (means, quats, scales (log), opacities (logit), sh0) of every splat is a piecewise-linear
curve with K control points (hat functions on K uniform knots, K fixed per run via E7_K). Higher SH bands (shN) are
static: trained in the section where the splat is born, frozen afterwards (the codec sends them once, at birth).

  * section 0: frame-0 fit exactly like e3 (back-projected depth, SH3, DefaultStrategy), all K knots = frame-0 values.
  * section g>0: warm start from the end state of g-1; the first knot of a continuing splat IS the last knot of g-1
    (shared boundary knot, gradient masked); the other knots start at that value.
  * training: per step E7_FB random frames of the section x E7_CB random train cameras, curve evaluated at t (only
    splats alive at t are rendered), loss 0.8 L1 + 0.2 D-SSIM (e3.loss_fn).
  * lifetimes like v3: the section is split into refresh windows of E7_REFRESH frames starting at r_w. Image-plane
    gradients are accumulated per window; at each refine event a high-gradient splat alive in window w is
      - cloned (small): a newborn with birth r_w and the same curve,
      - split (large): it dies at r_w and two smaller children (curve offsets per knot) are born at r_w,
    and a splat whose opacity curve stays below E7_PRUNE from r_w to the end of the section dies at r_w.
    Earlier frames of the section are never touched by these ops (no jumps inside a curve).
Knot placement E7_KNOTS: 'gop' (default) = knots on frames g0 .. g0+L-1 like e6's hat basis, so e6 measures the
trained K directly; the shared knot then holds the value for one frame at each section boundary. 'overlap' = knots on
g0 .. g0+L (the last knot sits on the first frame of the next section; no hold, but not aligned with e6).

Output per K: <RUN>/k<K>/tracks.npz (same format as e3 v3: per-frame values sampled from the curves over the global id
space, NaN when not alive; birth, death, shN), <RUN>/k<K>/curves.npz (control points per section), <RUN>/k<K>/e7.json
(PSNR / fg-PSNR per frame, alive, born, died, timings), previews; <RUN>/e7.json = summary over all K in RUN.
The frame-0 fit is cached in <RUN>/init.pt and reused by every K of the same RUN.
"""
import glob, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext'))
if os.environ.get('P8_GT'):    # matted rig scenes: GT dir from $P8_GT, W x H, visual-hull init
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rig'))
    import rig_track as e3
else:
    import e3_track as e3      # cameras, GT path, renderer, metrics, init_points, fit_densify (no training on import)
import curves_ext as cx    # P16: adaptive K, boundary smoothness
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat.utils import normalized_quat_to_rotmat

SMOKE = e3.SMOKE
dev = e3.dev
TRACKED = e3.TRACKED
T = 4 if SMOKE else int(os.environ.get('E7_FRAMES', e3.meta['frames']))
L = int(os.environ.get('E7_L', '2' if SMOKE else '32'))
KS = [int(x) for x in os.environ.get('E7_K', '2,3' if SMOKE else '5').split(',')]
ITS = int(os.environ.get('E7_ITS', '200' if SMOKE else '15000'))
REFRESH = int(os.environ.get('E7_REFRESH', '1' if SMOKE else '8'))
FB = int(os.environ.get('E7_FB', '2'))           # frames per step
CB = int(os.environ.get('E7_CB', '2'))           # train cameras per frame
RE = int(os.environ.get('E7_RE', '50' if SMOKE else '300'))      # refine every
RS = int(os.environ.get('E7_RS', '50' if SMOKE else '500'))      # refine start
RSTOP = float(os.environ.get('E7_RSTOP', '0.6'))                 # refine stop (fraction of ITS)
GROW = float(os.environ.get('E7_GROW', '0.0002'))                # grad2d threshold (gsplat default)
PRUNE = float(os.environ.get('E7_PRUNE', '0.002'))               # opacity threshold (v3 refresh)
KNOTS = os.environ.get('E7_KNOTS', 'gop')
SOFT0 = os.environ.get('E7_SOFT_KNOT0', '0') == '1'             # P2: knot 0 of continuing splats trainable (no shared knot)
EDGE_W = float(os.environ.get('E7_EDGE_W', '1'))                 # P2: sampling weight of the first/last EDGE_N frames
EDGE_N = int(os.environ.get('E7_EDGE_N', '4'))
CURV = float(os.environ.get('E7_CURV', '0'))                    # P2: L1 on knot second differences (per quant step) so
                                                                # that the encoder's K reduction (codec/prep --ktol) finds
                                                                # collinear knots; 0 = off
CURV_ATTRS = os.environ.get('E7_CURV_ATTRS', 'means,quats,scales,opacities,sh0').split(',')
SPAT = float(os.environ.get('E7_SPAT', '0'))                    # S2: L1 between a splat's knot differences and
                                                                # the mean knot differences of its SPAT_K nearest
                                                                # neighbours (knot 0 means), per quant step; makes the
                                                                # motion locally coherent so the codec's k-NN prediction
                                                                # (.csgs v3 PRED_SPATIAL) leaves small residuals; 0 = off
SPAT_K = int(os.environ.get('E7_SPAT_K', '6'))
SPAT_ATTRS = os.environ.get('E7_SPAT_ATTRS', 'means,quats').split(',')
SPAT_EVERY = int(os.environ.get('E7_SPAT_EVERY', '500'))         # neighbour search interval (steps)
SEED = int(os.environ.get('E7_SEED', '0'))
TRACKS = os.environ.get('E7_TRACKS', 'full')                    # P10 long runs: 'thin' = tracks.npz without the dense [T, N]
                                                                # per-frame arrays (640 frames x 600k ids would not fit in RAM);
                                                                # keeps birth/death/shN and means as the per-frame [min, max]
                                                                # envelope [T, 2, 3], which is all codec.prep/e8dec read from it
BASIS = os.environ.get('E7_BASIS', 'hat')                       # P10 ablation: 'bspline' = clamped cubic B-spline over the
                                                                # K control points (NOT decodable by .csgs, float quality only)
# ---- P13 switches, all off by default = P2 behaviour --------------------------------------------------------
# Diagnosis on the reference scene: along a section the opacity knots sink monotonically under gsplat's 1/255 alpha cull
# (section 0: 11 % of the splats at knot 0 -> 48 % at knot 4; c31 head cohort 11 % -> 71 %), the splat then gets no
# gradient any more ("zombie", P10) and is pruned at the next section start (20.7 k deaths at f32). Back of the head /
# hair at c31 thin out from ~f24 for exactly that reason.
OPA_LR = float(os.environ.get('E7_OPA_LR', '2e-2'))              # Adam lr of the opacity knots (P2: 2e-2)
OPA_FLOOR = os.environ.get('E7_OPA_FLOOR', '')                   # logit floor for opacity knots, e.g. -5.4 (alpha 0.0045 >
                                                                # cull 1/255): a knot can never sink below the cull, so the
                                                                # loss keeps deciding; combine with E7_PRUNE above the floor
OPA_FLOOR = float(OPA_FLOOR) if OPA_FLOOR else None
OPA_DECL = float(os.environ.get('E7_OPA_DECL', '0'))              # P13: weight of the hinge penalty on opacity decline between knots (0 = off)
OPA_DECL_MIN = float(os.environ.get('E7_OPA_DECL_MIN', '0.1'))    # ... only for splats with alpha > this at the section start
OPA_K1 = os.environ.get('E7_OPA_K1', '0') == '1'                 # one opacity per splat and section (knots tied -> K=1 in
                                                                # the codec): no fade inside a section, lifetimes only
CURRIC = float(os.environ.get('E7_CURRIC', '0'))                 # curriculum over the section: during the first CURRIC x ITS
                                                                # iterations the sampled frame window grows linearly from
                                                                # g0 to g1-1, and every knot beyond the first knot past the
                                                                # window end is tied to that knot (:= its value after each
                                                                # step). Knots therefore start from the converged value of
                                                                # their predecessor (warm start like e3 tracking) instead of
                                                                # all from the frame-g0 state, so fast parts (hair) are not
                                                                # faded out because knot 4 never found its target. 0 = off
# occlusion persistence, same switches/defaults as the rig trainer: a splat
# that was clearly visible and is now covered by nearer opaque splats (depth test over train cams) goes dormant instead
# of drifting/fading/dying: its knots in the occluded windows get no gradient, rotation/scale/opacity/colour hold the last
# non-dormant knot, position follows the mean motion of its FOLLOW_K nearest non-dormant neighbours (codec csgs v2
# 'follow': no motion sent). Only visible-but-useless splats are pruned additionally ("superfluous").
DORM = os.environ.get('E7_DORMANT', '0') == '1'
DORM_FRAC = float(os.environ.get('E7_DORM_FRAC', '0.05'))     # occluded: contribution < FRAC x the splat's best so far
DORM_MINC = float(os.environ.get('E7_DORM_MINC', '0.5'))      # ... and the best so far >= MINC pixel-weight per view
DORM_BEH = float(os.environ.get('E7_DORM_BEH', '0.5'))       # ... and behind the surface in >= BEH of its in-image views
DORM_TOL = float(os.environ.get('E7_DORM_TOL', '0.03'))       # depth tolerance (scene units) of that test
DORM_SUPC = float(os.environ.get('E7_DORM_SUPC', '0.01'))     # superfluous: in front, in the image, < SUPC weight per view
DORM_MAX = int(os.environ.get('E7_DORM_MAX', '0'))            # >0: die after that many whole sections dormant (owner: off)
FOLLOW_K = 4                                                  # = codec.falt.FOLLOW_K
DORM_CAMSTEP = int(os.environ.get('E7_DORM_CAMSTEP', '2'))
DORM_DECAY = float(os.environ.get('E7_DORM_DECAY', '1'))      # best-so-far contribution decays per window
# P13 under-reconstruction: the floor run showed that the head region is short of splats towards the section
# end (too few / too late births), the silent deaths are a symptom. Generic densification levers, all off by default:
#  * E7_ERRGROW=1: error-driven births. At every refine event and refresh window, E7_ERR_CAMS train views of one frame
#    of the window (frame rotates with the event) are rendered; pixel error e = mean_rgb |render - GT| on GT-foreground
#    pixels (GT non-black: generic, no region mask), box-averaged over E7_ERR_TILE px, excess e' = relu(e - E7_ERR_T).
#    Per splat the error mass m_i = sum_p e'(p) w_i(p) (w = alpha*T rendering weight, one extra render+backward of a
#    ones colour). Holes (GT foreground, rendered alpha < 0.5) have no splat weight: their excess error is assigned to
#    the nearest visible splat centre in 2D (E7_ERR_HOLE=1). The top E7_ERR_N splats by mass (m >= E7_ERR_MIN) per
#    window are cloned/split by the existing refine path, independent of the gradient threshold.
#  * E7_ABSGRAD=1: densify on gsplat's absgrad (sum of |per-pixel grads|, AbsGS) instead of |sum|; raise E7_GROW with it.
#  * E7_RSTOP (existing) for longer densification within the section.
ERRGROW = os.environ.get('E7_ERRGROW', '0') == '1'
ERR_T = float(os.environ.get('E7_ERR_T', '0.05'))
ERR_N = int(os.environ.get('E7_ERR_N', '200'))                 # anchor grows only ~8k splats per section (P13 floor log)
ERR_MIN = float(os.environ.get('E7_ERR_MIN', '0.5'))
ERR_CAMS = int(os.environ.get('E7_ERR_CAMS', '6'))
ERR_TILE = int(os.environ.get('E7_ERR_TILE', '4'))
ERR_HOLE = os.environ.get('E7_ERR_HOLE', '1') == '1'
ERR_OPA = os.environ.get('E7_ERR_OPA', '')                     # optional: opacity-logit floor of error-driven newborns
ERR_OPA = float(ERR_OPA) if ERR_OPA else None
ABSGRAD = os.environ.get('E7_ABSGRAD', '0') == '1'
# P13 relocation (owner priority after the first sweep): E7_RELOC=1 moves silently dead splats (opacity knots below
# E7_RELOC_T from some knot to the section end) onto targets (MCMC-style, see relocate()); every E7_RE its from E7_RS
# until E7_RELOC_STOP x ITS (default = E7_RSTOP). E7_RELOC_W = opa | err | opaerr (target weights), E7_RELOC_MAX caps
# the moves per knot class and event (0 = no cap).
RELOC = os.environ.get('E7_RELOC', '0') == '1'
RELOC_T = float(os.environ.get('E7_RELOC_T', '0.005'))
RELOC_W = os.environ.get('E7_RELOC_W', 'opa')
RELOC_MAX = int(os.environ.get('E7_RELOC_MAX', '0'))
RELOC_STOP = float(os.environ.get('E7_RELOC_STOP', os.environ.get('E7_RSTOP', '0.6')))
RELOC_HIDE = os.environ.get('E7_RELOC_HIDE', '1') == '1'       # move while invisible: the first replaced knot keeps the
                                                                # dead opacity (0 = first sweep 'rel': fade-in while flying)
# owner rule "areas that need more detail get more splats": E7_ERR_DETAIL=1 weights the excess error of the error map
# (births AND relocation targets) by the GT's frequency content (|Laplacian| of the grey GT, 8-px tile mean, normalised
# to mean 1 on the GT foreground). Generic, no region rule.
ERR_DETAIL = os.environ.get('E7_ERR_DETAIL', '0') == '1'
# P13 finding: the c31 head is 41.7 dB in the nearest TRAIN view (c32, 26.5 deg away) but 26.4 dB in c31 -> not
# under-reconstruction but a held-out-view gap (dark splats/streaks over the hair between the views). 1.9 % of the
# (eroded) black GT background of the train views is covered by opaque splats (probe_bg_alpha.py): against black they are
# free. E7_BGALPHA=w adds w x mean rendered alpha over the eroded GT background (GT max channel < 2/255, -3 px) to the
# loss. Generic (no region rule); for vendor GT with a black background the GT itself is the matte.
BGALPHA = float(os.environ.get('E7_BGALPHA', '0'))
FGALPHA = float(os.environ.get('E7_FGALPHA', '0'))              # ... and w x mean (1 - alpha) over the eroded GT foreground
                                                                # (thin/fading hair lets interior splats show in new views)
# ---- P17 switches, all off by default = P13 behaviour ------------------------------------------------------
# Higher-resolution reference with a second, closer camera ring (GT dir ..._2c1024n;
# camera field 'ring' = far | near). Generic levers, no region rule:
#  * E7_CAMS=far: train only on the far (P13) ring of such a GT (ablation "resolution without the new views").
#  * E7_LOWRES=s, E7_LOWRES_FRAC=f: resolution plan, the first f x ITS steps of every section render at scale s (GT
#    area-downsampled on the fly, intrinsics scaled), the rest at full GT resolution. E7_INIT_SCALE=s: frame-0 fit at s.
#  * E7_CAP=n: hard cap on the splat rows of a section pool; when growth would exceed it, the candidates with the highest
#    (detail-weighted) gradient score win (budget moves to where it is needed most).
#  * E7_DETAIL=freq|errfreq: detail-adaptive densification. At every refine event, per refresh window and splat a detail
#    weight D_i = sum_p w_i(p) d(p) / sum_p w_i(p) over E7_DETAIL_CAMS train views (w = rendering weight), with d the GT
#    frequency content (|Laplacian| of grey GT, tile mean) [x the local reconstruction error for errfreq], normalised to
#    mean 1 on the GT foreground; the grow score is grad x clamp(D, 1/4, 4)^E7_DETAIL_GAMMA. Splats in detail-poor,
#    well-reconstructed areas grow less, detailed/erroneous areas more (combined with E7_CAP the total stays bounded).
#  * E7_DEPTH=w: depth guidance, w x mean |ED - D_gt| / D_gt over GT-foreground pixels (gsplat expected depth vs the
#    reference depth c##_d16.png of every frame; GT dirs without them fall back to no depth term with a warning).
CAMSEL = os.environ.get('E7_CAMS', 'all')
LOWRES = float(os.environ.get('E7_LOWRES', '1'))
LOWRES_FRAC = float(os.environ.get('E7_LOWRES_FRAC', '0'))
INIT_SCALE = float(os.environ.get('E7_INIT_SCALE', '1'))
CAP = int(os.environ.get('E7_CAP', '0'))
DETAIL = os.environ.get('E7_DETAIL', '')
DETAIL_CAMS = int(os.environ.get('E7_DETAIL_CAMS', '6'))
DETAIL_GAMMA = float(os.environ.get('E7_DETAIL_GAMMA', '1'))
DEPTH = float(os.environ.get('E7_DEPTH', '0'))
MIXRES = float(os.environ.get('E7_MIXRES', '0'))               # P17: after the low-res phase, each step renders at
                                                                # E7_LOWRES with this probability (multi-scale: a model
                                                                # fitted only at 1024 aliases/thickens when shown at 512)
# ---- P16 switches, all off by default = P13 behaviour; helpers in train/ext/curves_ext.py --------------
# E7_BASIS=bspline is decodable now (csgs v6 / flag 64): curves.npz carries basis='bspline', codec.measure --basis auto reads it.
# Adaptive K (E7_KADAPT=1): E7_K = columns allocated (e.g. 9), every splat starts a section with E7_KLO (5) of them; at
# E7_KADAPT_AT x ITS (after the densify phase, so the densify code sees a plain K_lo run) the top E7_KADAPT_FRAC of the visible
# splats by motion score (E7_KADAPT_SCORE = path | curv: control polygon length | sum of 2nd differences of the K_lo curve)
# get all E7_K columns (refitted to their K_lo curve) and are trained on with them.
KADAPT = os.environ.get('E7_KADAPT', '0') == '1'
KLO = int(os.environ.get('E7_KLO', '5'))
KADAPT_FRAC = float(os.environ.get('E7_KADAPT_FRAC', '0.15'))
KADAPT_AT = float(os.environ.get('E7_KADAPT_AT', '0.6'))
KADAPT_SCORE = os.environ.get('E7_KADAPT_SCORE', 'path')
# Smoother section transitions (E7_SMOOTH=w): continuing splats get a penalty (per quantisation step, L1, weight w) for the
# jump between the previous section's end and this section's start: first control point vs the previous end value
# extrapolated by one frame (end velocity), and start velocity vs end velocity. Format-neutral (encoder-side only).
SMOOTH = float(os.environ.get('E7_SMOOTH', '0'))
SMOOTH_ATTRS = os.environ.get('E7_SMOOTH_ATTRS', 'means,quats').split(',')
# E7_CURV_GREVILLE=1 (with E7_BASIS=bspline): the curvature term E7_CURV measures deviation from motion linear IN TIME (control
# points on the Greville abscissae) instead of plain second differences of the control polygon. With the plain term a clamped
# B-spline whose control polygon is collinear and uniformly spaced still accelerates (slow ends), so the codec's K reduction
# (ktol) finds fewer exact reductions than for the hat basis (P16 finding: B-spline stream +12 % Mbit/s).
CURV_GREV = os.environ.get('E7_CURV_GREVILLE', '0') == '1'
# P22: temporal colour smoothing where the ORIGINAL does not change colour (no sharpness loss, no skin rule).
# E7_CSMOOTH=w: per splat and section the "real change" r = RMS deviation over the section of the GT colour sampled ALONG the
# splat's own track (projected centre in every training camera, GT box-filtered with E7_CSMOOTH_BOX px, only frames where the
# splat is the visible surface: depth <= rendered expected depth x (1 + E7_CSMOOTH_DTOL) and accumulated alpha > 0.5; per
# camera >= E7_CSMOOTH_MINF frames, weighted mean over cameras). Weight w_i = clamp((HI - r) / (HI - LO), 0, 1) (r <= LO: no
# real change -> 1, r >= HI: real change (light, shadow, folds) -> 0; never seen -> E7_CSMOOTH_UNSEEN). Loss
# w x w_i x |knot - mean of the splat's knots| / quantisation step (L1: pulls static-colour curves to ONE value, i.e. K=1 in
# the codec) for E7_CSMOOTH_ATTRS, + the same weight on |knot 0 - previous section's end value| for continuing splats
# (continuity over section boundaries -> difference 0). Geometry, shN, scales untouched. Weights at the section start and
# every E7_CSMOOTH_EVERY steps (0 = start only; densify adds rows with weight 0 until the next update). Default off.
CSMOOTH = float(os.environ.get('E7_CSMOOTH', '0'))
CSM_ATTRS = [a for a in os.environ.get('E7_CSMOOTH_ATTRS', 'sh0').split(',') if a]
CSM_LO = float(os.environ.get('E7_CSMOOTH_LO', '0.01'))
CSM_HI = float(os.environ.get('E7_CSMOOTH_HI', '0.03'))
CSM_BOX = int(os.environ.get('E7_CSMOOTH_BOX', '5'))
CSM_DTOL = float(os.environ.get('E7_CSMOOTH_DTOL', '0.02'))
CSM_MINF = int(os.environ.get('E7_CSMOOTH_MINF', '4'))
CSM_UNSEEN = float(os.environ.get('E7_CSMOOTH_UNSEEN', '0'))
CSM_EVERY = int(os.environ.get('E7_CSMOOTH_EVERY', '0'))
CSM_MOVE = float(os.environ.get('E7_CSMOOTH_MOVE', '5'))           # report only: "moving" = path length > MOVE x 1e-3 x ext
CSM_MODE = os.environ.get('E7_CSMOOTH_MODE', 'mean')              # mean: L1 to the knot mean (K=1); d2: weighted 2nd differences
# P22: E7_TDIFF=w -> every step renders an ADJACENT frame pair (t, t+1) with the same cameras and adds
# w x mean |(R_t+1 - R_t) - (G_t+1 - G_t)| (the P10 flicker index as a loss: temporal changes must match the GT's; no blur,
# no region rule). Default 0 = unchanged random frames.
TDIFF = float(os.environ.get('E7_TDIFF', '0'))
# P22: E7_BOUND=w -> continuing splats get w x mean |knot 0 - previous section's end value| / quantisation step for
# E7_BOUND_ATTRS (value continuity over the section boundary, also in E7_FINETUNE where E7_SMOOTH has no previous state).
BOUND = float(os.environ.get('E7_BOUND', '0'))
BOUND_ATTRS = [a for a in os.environ.get('E7_BOUND_ATTRS', 'opacities,means,scales').split(',') if a]
# P24, see train/flow/flowguide.py: E7_FLOW=w binds the projected 2D motion of the splats (from the curves,
# frame s -> s+1) to the precomputed GT optical flow of E7_FLOW_DIR (confidence- and edge-weighted, E7_FLOW_EDGE);
# E7_MSAMP=a samples the frames of a section by GT motion energy, E7_TAIL_W/E7_TAIL_N weight the last frames. Default off.
FLOW = float(os.environ.get('E7_FLOW', '0'))
FLOW_DIR = os.environ.get('E7_FLOW_DIR', '')
FLOW_EDGE = float(os.environ.get('E7_FLOW_EDGE', '0.75'))
MSAMP = float(os.environ.get('E7_MSAMP', '0'))
TAIL_W = float(os.environ.get('E7_TAIL_W', '1'))
TAIL_N = int(os.environ.get('E7_TAIL_N', '8'))
if FLOW > 0 or MSAMP > 0 or TAIL_W != 1:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flow'))
    import flowguide as fgd
PREV_ATTRS = list(dict.fromkeys((CSM_ATTRS if CSMOOTH > 0 else []) + (BOUND_ATTRS if BOUND > 0 else [])))
# P36, all off by default = unchanged behaviour:
#  * E7_ITS0=n: section 0 trains n steps instead of E7_ITS (longer first section; the init loses ~2 dB on flame there).
#  * E7_STATIC_LR=m: "static" splats (centre at the section start inside GT pixels that do not change over the section
#    in every train camera that sees it) move only m x their Adam update for the tracked attributes (+ shN), so the
#    background keeps the frame-0 fit. Static pixels: per train camera max_t |GT_t - GT_g0| (RGB mean, 5x5 box) <=
#    E7_STATIC_T, the moving region dilated by E7_STATIC_DIL px. Re-classified at the section start and after each refine.
#  * E7_STATIC_TIE=1: static splats carry ONE value over the section (all K knots := their mean after every step), i.e.
#    the static part is trained from all frames x cameras of the section and costs no motion knots (P36 stage 4).
ITS0 = int(os.environ.get('E7_ITS0', '0'))
STATIC_LR = float(os.environ.get('E7_STATIC_LR', '1'))
STATIC_TIE = os.environ.get('E7_STATIC_TIE', '0') == '1'
STATIC_T = float(os.environ.get('E7_STATIC_T', '0.03'))
STATIC_DIL = int(os.environ.get('E7_STATIC_DIL', '9'))
STATIC_ON = STATIC_LR != 1 or STATIC_TIE
#  * E7_PROBE0=n (diagnosis of the frame-0 loss, stage 3): cam00 PSNR (and mean train-cam PSNR) of the FIRST frame of the
#    section before training, every n steps, and right before / after every refine event -> ev['probe0'] = [[it, tag,
#    test PSNR, train PSNR, N], ...]. Renders only, no effect on training (default 0 = off).
PROBE0 = int(os.environ.get('E7_PROBE0', '0'))
#  * E7_LRDECAY_ALL=f (stage 3): the learning rates of ALL attributes (not only the means) decay exponentially to f x
#    their start value over the section (diagnosis: with constant colour/opacity lr the cam00 PSNR of frame 0 swings by
#    0.3-3 dB from step to step late in the section, the end value is a sample of that swing). Default 0 = off (only the
#    means decay to 0.1, unchanged).
LRDECAY_ALL = float(os.environ.get('E7_LRDECAY_ALL', '0'))
# ----------------------------------------------------------------------------------------------------------------------
RUN = os.environ.get('E7_RUN', os.path.join(e3.ROOT, 'runs', 'e7', time.strftime('%Y%m%d-%H%M%S')))
# S1: retraining with fixed SN. E7_FINETUNE=<k_dir> loads that run's curves.npz/tracks.npz section by section
# (same ids/lifetimes, no init fit, no densify/prune) and continues training all knots for E7_ITS steps; E7_SN_FIXED=
# <assignment.npz> (codec.snvq.Assignment) replaces shN by the SN-VQ reconstruction and freezes it. Default: off (P1 path).
FINETUNE = os.environ.get('E7_FINETUNE', '')
SN_FIXED = os.environ.get('E7_SN_FIXED', '')
LR_SCALE = float(os.environ.get('E7_LR_SCALE', '1'))             # S1: scale of lr_table (finetune uses a smaller one)
BIG = 1 << 30
W, H = e3.W, e3.H
W0, H0, K0 = e3.W, e3.H, e3.K.clone()                            # P17: full GT resolution and intrinsics
SCALE = 1.0
if CAMSEL == 'far':                                              # P17: in place, so e3.init_points/fit_densify see it too
    e3.train[:] = [i for i in e3.train if e3.cams[i].get('ring', 'far') == 'far']
elif CAMSEL != 'all':
    raise SystemExit(f'E7_CAMS={CAMSEL}: all | far')


def set_res(s):
    """P17: render at scale s of the GT resolution (e3.render, render_train, contributions read these globals)."""
    global W, H, SCALE
    SCALE = s
    W, H = int(round(W0 * s)), int(round(H0 * s))
    e3.W, e3.H = W, H
    Ks = K0.clone()
    Ks[:, :2, :] *= s
    e3.K = Ks


def gt_now(x):
    """GT images [..., H0, W0, 3] (uint8 or float) -> float [..., H, W, 3] in [0, 1] at the current render scale."""
    x = x.float() / 255.0 if x.dtype == torch.uint8 else x
    if SCALE == 1.0:
        return x
    sh = x.shape
    y = Fn.interpolate(x.reshape((-1,) + sh[-3:]).permute(0, 3, 1, 2), size=(H, W), mode='area')
    return y.permute(0, 2, 3, 1).reshape(sh[:-3] + (H, W, 3))


def depth_now(d):
    """GT depth [C, H0, W0] (float, 0 = background) -> [C, H, W] at the current scale (mean over valid pixels)."""
    if SCALE == 1.0:
        return d
    v = (d > 0).float()[:, None]
    num = Fn.interpolate(d[:, None] * v, size=(H, W), mode='area'); den = Fn.interpolate(v, size=(H, W), mode='area')
    return torch.where(den > 0.5, num / den.clamp_min(1e-6), torch.zeros_like(num))[:, 0]


def load_depth(g0, g1):
    """P17: uint16 GT depth of frames [g0, g1) on the CPU [Lg, C, H, W] (None if the GT dir has no per-frame depth)."""
    if not os.path.exists(f'{e3.GT}/f{g0:03d}/c00_d16.png'):
        print('WARNING: E7_DEPTH set but no c##_d16.png in the GT dir -> no depth term', flush=True)
        return None
    out = torch.empty((g1 - g0, len(e3.cams), H0, W0), dtype=torch.int16)
    for t in range(g0, g1):
        for i in range(len(e3.cams)):
            a = np.asarray(Image.open(f'{e3.GT}/f{t:03d}/c{i:02d}_d16.png')).astype(np.uint16)
            out[t - g0, i] = torch.from_numpy(a.view(np.int16).copy())                # bit pattern, read back as uint16
    return out


def depth_of(dep, t, g0, cams):
    d = dep[t - g0][cams].to(dev).to(torch.int32) & 0xFFFF
    return depth_now(d.float() / float(e3.meta.get('depth_scale', 16384.0)))


def load_section(g0, g1):
    """uint8 GT of frames [g0, g1) on the GPU: [Lg, C, H, W, 3] (32 frames x 36 cams at 512^2 = 0.9 GB)."""
    out = torch.empty((g1 - g0, len(e3.cams), H, W, 3), dtype=torch.uint8, device=dev)
    for t in range(g0, g1):
        for i in range(len(e3.cams)):
            out[t - g0, i] = torch.from_numpy(np.asarray(Image.open(f'{e3.GT}/f{t:03d}/c{i:02d}.png'))[..., :3].copy())
    return out


def span_of(Lg):
    return max((Lg - 1) if KNOTS == 'gop' else Lg, 1)


def knot_at(t, g0, Lg, K):
    """segment index j and weight w of knot j+1 for frame t."""
    tau = (t - g0) * (K - 1) / span_of(Lg)
    j = min(int(math.floor(tau)), K - 2)
    return j, tau - j


def bspline_weights(K, u, p=3):
    """clamped B-spline basis of degree min(p, K-1) with K control points, uniform interior knots, at u in [0, 1]
    (Cox-de Boor; pure Python, K is small). Sums to 1; u=0 -> control point 0, u=1 -> control point K-1."""
    p = min(p, K - 1)
    if p == 0:
        return [1.0]
    n_int = K - p - 1
    kn = [0.0] * (p + 1) + [(i + 1) / (n_int + 1) for i in range(n_int)] + [1.0] * (p + 1)
    u = min(max(float(u), 0.0), 1.0)
    m = len(kn) - 1
    N = [1.0 if (kn[i] <= u < kn[i + 1]) or (u == 1.0 and kn[i] < kn[i + 1] == 1.0) else 0.0 for i in range(m)]
    for d in range(1, p + 1):
        N = [((u - kn[i]) / (kn[i + d] - kn[i]) * N[i] if kn[i + d] > kn[i] else 0.0) +
             ((kn[i + d + 1] - u) / (kn[i + d + 1] - kn[i + 1]) * N[i + 1] if kn[i + d + 1] > kn[i + 1] else 0.0)
             for i in range(m - d)]
    return N[:K]


def curve_at(P, idx, t, g0, Lg, K):
    if BASIS == 'bspline' and K > 2:
        w = bspline_weights(K, (t - g0) / span_of(Lg))
        nz = [j for j in range(K) if w[j] != 0.0]
        q = {}
        for k in TRACKED:
            acc = w[nz[0]] * P[k][idx, nz[0]]
            for j in nz[1:]:
                acc = acc + w[j] * P[k][idx, j]
            q[k] = acc
        q['sh0'] = q['sh0'][:, None]
        q['shN'] = P['shN'][idx]
        return q
    j, w = knot_at(t, g0, Lg, K)
    q = {k: (1 - w) * P[k][idx, j] + w * P[k][idx, j + 1] for k in TRACKED}
    if OPA_K1:                                                    # P13: tied opacity knots, knot 0 carries the gradient
        q['opacities'] = P['opacities'][idx, 0]
    q['sh0'] = q['sh0'][:, None]
    q['shN'] = P['shN'][idx]
    return q


def alive_idx(S, t):
    return torch.nonzero((S['birth'] <= t) & (S['death'] > t)).squeeze(1)


@torch.no_grad()
def knn_indices(x, k, chunk=1024):
    """[N, k] indices of the k nearest OTHER points of x [N, 3] (chunked cdist; N < k+1 -> repeats of self)."""
    N = x.shape[0]
    if N <= k:                                                   # degenerate (tiny pools): repeat the indices
        if N == 0:
            return torch.zeros((0, k), dtype=torch.long, device=x.device)
        return torch.arange(N, device=x.device).repeat((k + N - 1) // N)[:k][None].expand(N, k).contiguous()
    out = torch.empty((N, k), dtype=torch.long, device=x.device)
    for a in range(0, N, chunk):
        d = torch.cdist(x[a:a + chunk], x)
        d[torch.arange(d.shape[0], device=x.device), torch.arange(a, a + d.shape[0], device=x.device)] = float('inf')
        out[a:a + chunk] = d.topk(k, dim=1, largest=False).indices
    return out


def spat_loss(P, nn, steps):
    """SPAT term: mean over attrs, splats, knot intervals and dims of |diff_i - mean_j diff_j| / step."""
    loss = 0.0
    for k, st in steps.items():
        v = P[k]
        if v.shape[1] < 2:
            continue
        dlt = v[:, 1:] - v[:, :-1]                                   # [N, K-1, d]
        nb = dlt[nn].mean(1)                                         # [N, k, K-1, d] -> [N, K-1, d]
        loss = loss + ((dlt - nb).abs() / st).mean()
    return loss

def render_train(q, cams, depth=False):
    """training render; = e3.render unless E7_ABSGRAD (then gsplat also accumulates |per-pixel grads| in absgrad) or
    depth (P17: render_mode RGB+ED, returns (rgb, alpha, info, expected depth [C,H,W]))."""
    if not ABSGRAD and not depth:
        return e3.render(q, cams)
    from gsplat import rasterization
    col = torch.cat([q['sh0'], q['shN']], 1) if q['shN'].shape[1] else q['sh0']
    deg = int(math.isqrt(col.shape[1]) - 1)
    rgb, alpha, info = rasterization(q['means'], q['quats'], torch.exp(q['scales']), torch.sigmoid(q['opacities']),
                                     col, e3.VM[cams], e3.K[cams], W, H, sh_degree=deg, packed=False, absgrad=ABSGRAD,
                                     render_mode='RGB+ED' if depth else 'RGB')
    if depth:
        return rgb[..., :3].clamp(0, 1), alpha, info, rgb[..., 3]
    return rgb.clamp(0, 1), alpha, info


def lr_table(ext):
    base = {'means': 1e-4 * ext, 'quats': 1e-3, 'scales': 3e-3, 'opacities': OPA_LR, 'sh0': 2e-3, 'shN': 2.5e-3 / 20}
    return {k: v * LR_SCALE for k, v in base.items()}


@torch.no_grad()
def rebuild(P, opts, S, keep, adds, add_meta):
    """keep rows `keep` (LongTensor), append rows `adds` (dict per param) with meta add_meta; Adam state of new rows = 0.
    Extra per-splat state in S (P13 dormant: cmax, dk, nb, dage) is kept/zero-padded like p8_e7."""
    n_add = len(add_meta['birth']) if add_meta else 0

    def pf(name, p):
        v = p[keep]
        if n_add:
            v = torch.cat([v, adds[name]])
        return torch.nn.Parameter(v.contiguous(), requires_grad=p.requires_grad)

    def of(key, v):
        return torch.cat([v[keep], torch.zeros((n_add,) + v.shape[1:], device=v.device, dtype=v.dtype)])

    _update_param_with_optimizer(pf, of, P, opts)
    for k in list(S):
        v = S[k][keep]
        if n_add:
            v = torch.cat([v, add_meta[k] if k in add_meta else torch.zeros((n_add,) + v.shape[1:], dtype=v.dtype, device=v.device)])
        S[k] = v


@torch.no_grad()
def split_rows(P, rows):
    """two children per row: offsets sampled once in the local frame and mapped through each knot's rotation/scale."""
    n, K = len(rows), P['means'].shape[1]
    q = Fn.normalize(P['quats'][rows], dim=-1)
    R = normalized_quat_to_rotmat(q.reshape(-1, 4)).reshape(n, K, 3, 3)
    s = torch.exp(P['scales'][rows])
    z = torch.randn(2, n, 1, 3, device=dev).expand(2, n, K, 3)
    off = torch.einsum('nkij,nkj,bnkj->bnki', R, s, z)
    out = {}
    for k in P:
        v = P[k][rows]
        if k == 'means':
            out[k] = (v[None] + off).reshape(2 * n, K, 3)
        elif k == 'scales':
            out[k] = torch.log(s / 1.6).repeat(2, 1, 1)
        else:
            out[k] = v.repeat((2,) + (1,) * (v.dim() - 1))
    return out


@torch.no_grad()
def refine(P, opts, S, gsum, gcnt, g0, Lg, K, scene_scale, extra=None, dw=None):
    """grow (clone/split) per refresh window, then prune; returns (cloned, split, pruned).
    extra: optional [nW, N] bool (P13 error-driven selection), grown in addition to the gradient threshold.
    dw: optional [nW, N] detail weight multiplying the gradient score (P17 E7_DETAIL); E7_CAP bounds the pool rows."""
    birth, death = S['birth'], S['death'].clone()
    N = len(birth)
    taken = torch.zeros(N, dtype=torch.bool, device=dev)
    smax = torch.exp(P['scales']).amax(-1)                       # [N,K]
    adds, metas, n_cl, n_sp = [], [], 0, 0
    for w in range(gsum.shape[0]):
        r = g0 + w * REFRESH
        j0 = knot_at(r, g0, Lg, K)[0]
        alive = (birth <= r) & (death > r)
        gm = gsum[w] / gcnt[w].clamp_min(1)
        if dw is not None:
            gm = gm * dw[w]
        hg = gm > GROW
        hi = alive & (hg | extra[w] if extra is not None else hg) & ~taken
        small = smax[:, j0:].amax(-1) <= 0.01 * scene_scale
        if CAP > 0:                                               # P17: best scores first until the cap is reached
            room = CAP - N - n_cl - 2 * n_sp
            cand = torch.nonzero(hi).squeeze(1)
            hi = torch.zeros_like(hi)
            if room > 0 and len(cand):
                order = cand[torch.argsort(gm[cand], descending=True)]
                cost = torch.where(small[order], 1, 2)
                hi[order[torch.cumsum(cost, 0) <= room]] = True
        taken |= hi
        cl = torch.nonzero(hi & small).squeeze(1)
        sp = torch.nonzero(hi & ~small).squeeze(1)
        if len(cl):
            adds.append({k: P[k][cl].clone() for k in P})
            if ERR_OPA is not None and extra is not None:            # error-only newborns start above the cull
                eo = (extra[w] & ~hg)[cl]
                adds[-1]['opacities'][eo] = adds[-1]['opacities'][eo].clamp_min(ERR_OPA)
            metas.append({'birth': torch.full_like(cl, r), 'death': death[cl].clone(), 'ids': torch.full_like(cl, -1)}
                         | ({'cmax': S['cmax'][cl].clone()} if 'cmax' in S else {}))
        if len(sp):
            adds.append(split_rows(P, sp))
            metas.append({'birth': torch.full((2 * len(sp),), r, dtype=torch.long, device=dev),
                          'death': death[sp].repeat(2), 'ids': torch.full((2 * len(sp),), -1, dtype=torch.long, device=dev)}
                         | ({'cmax': S['cmax'][sp].repeat(2)} if 'cmax' in S else {}))
            death[sp] = r                                         # parent ends where its children start
        n_cl += len(cl); n_sp += len(sp)
    op = torch.sigmoid(P['opacities'])                            # [N,K]
    n_pr = 0
    for w in range(gsum.shape[0]):
        r = g0 + w * REFRESH
        j0 = knot_at(r, g0, Lg, K)[0]
        pr = (birth <= r) & (death > r) & (op[:, j0:].amax(-1) < PRUNE)
        death[pr] = r; n_pr += int(pr.sum())
    S['death'] = death
    keep = torch.nonzero(birth < death).squeeze(1)                # rows never alive (born and killed at r) go away
    add = {k: torch.cat([a[k] for a in adds]) for k in P} if adds else {}
    meta = {k: torch.cat([m[k] for m in metas]) for k in metas[0]} if metas else None
    rebuild(P, opts, S, keep, add, meta)
    return n_cl, n_sp, n_pr


# ---- P13: occlusion persistence (verbatim port of p8_e7.py contributions/follow_means/dormant_update/prune_superfluous) --
def contributions(P, S, g0, Lg, K):
    """per refresh window and splat over every DORM_CAMSTEP-th train camera, 2 frames per window:
    c      mean rendering weight per view (sum over pixels of alpha*G*T = d render(ones) / d ones),
    n_in   views where the centre projects into the image (radius > 0),
    n_beh  of those, views where the splat lies behind the rendered surface: its depth > expected depth at its centre
           pixel + DORM_TOL while that pixel is opaque (alpha > 0.5), i.e. covered by nearer opaque splats."""
    from gsplat import rasterization
    nW = (Lg + REFRESH - 1) // REFRESH
    N = len(S['birth'])
    c = torch.zeros(nW, N, device=dev)
    n_in = torch.zeros(nW, N, device=dev); n_beh = torch.zeros(nW, N, device=dev)
    cams = e3.train[::DORM_CAMSTEP]
    for w in range(nW):
        r = g0 + w * REFRESH
        frs = sorted({min(r + 1, g0 + Lg - 1), min(r + REFRESH // 2 + 1, g0 + Lg - 1)})
        for t in frs:
            idx = alive_idx(S, t)
            with torch.no_grad():
                q = curve_at(P, idx, t, g0, Lg, K)
            ones = torch.ones((len(idx), 1), device=dev, requires_grad=True)
            for i in range(0, len(cams), 8):
                ch = cams[i:i + 8]
                img, alpha, info = rasterization(q['means'], q['quats'], torch.exp(q['scales']), torch.sigmoid(q['opacities']),
                                                 ones, e3.VM[ch], e3.K[ch], W, H, sh_degree=None, packed=False,
                                                 render_mode='RGB+ED')
                img[..., 0].sum().backward()
                with torch.no_grad():
                    rad = info['radii']
                    vis = ((rad > 0).all(-1) if rad.dim() == 3 else rad > 0)                     # [C,n]
                    uv = info['means2d'].round().long()
                    inside = vis & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
                    u = uv[..., 0].clamp(0, W - 1); v = uv[..., 1].clamp(0, H - 1)
                    ci = torch.arange(len(ch), device=dev)[:, None].expand_as(u)
                    ed = img[..., 1].detach()[ci, v, u]; ap = alpha[..., 0].detach()[ci, v, u]
                    beh = inside & (ap > 0.5) & (info['depths'] > ed + DORM_TOL)
                    n_in[w].index_add_(0, idx, inside.float().sum(0))
                    n_beh[w].index_add_(0, idx, beh.float().sum(0))
            c[w].index_add_(0, idx, ones.grad[:, 0] / (len(cams) * len(frs)))
    return c, n_in, n_beh


@torch.no_grad()
def follow_means(P, S):
    """dormant position knots follow their neighbours: for knot j with anchor a (last non-dormant knot below j)
    x_j = x_a + mean over the neighbours of (x_n,j - x_n,a); same rule as the codec's csgs apply_follow (v2)."""
    dk = S['dk']
    rows = torch.nonzero(dk.any(1)).squeeze(1)
    if len(rows) == 0:
        return
    K = dk.shape[1]
    ar = torch.arange(K, device=dev).expand(len(rows), K)
    a = torch.where(dk[rows], torch.zeros_like(ar), ar).cummax(1).values           # anchor index per knot
    x = P['means'].data
    nb = S['nb'][rows]                                                               # [r, FK], -1 = none
    ok = nb >= 0
    nbx = x[nb.clamp_min(0)]                                                         # [r, FK, K, 3]
    disp = nbx - torch.gather(nbx, 2, a[:, None, :, None].expand(-1, nb.shape[1], -1, 3))
    disp = (disp * ok[..., None, None]).sum(1) / ok.sum(1).clamp_min(1)[:, None, None]
    pred = torch.gather(x[rows], 1, a[..., None].expand(-1, -1, 3)) + disp
    m = dk[rows]
    xr = x[rows]
    xr[m] = pred[m]
    x[rows] = xr


@torch.no_grad()
def dormant_update(P, opts, S, g0, Lg, K, c, n_in, n_beh):
    """classify every alive splat per refresh window (visible / occluded -> dormant / out of view / superfluous -> prune);
    see p8_e7.dormant_update. Returns (stats, prune mask per window [nW, N])."""
    nW = c.shape[0]
    span = span_of(Lg)
    wj = [min(int(j * span / (K - 1)) // REFRESH, nW - 1) for j in range(K)]       # window of each knot
    cmax = S['cmax'].clone()
    N = len(cmax)
    occ = torch.zeros(nW, N, dtype=torch.bool, device=dev); sup = torch.zeros_like(occ)
    st = {'visible': 0, 'occluded': 0, 'out_of_view': 0, 'superfluous': 0}
    for w in range(nW):
        r = g0 + w * REFRESH
        alive = (S['birth'] <= r) & (S['death'] > r)
        vis = alive & (c[w] >= DORM_FRAC * cmax) & (c[w] > 0)
        cmax = torch.where(vis, torch.maximum(cmax * DORM_DECAY, c[w]), cmax * DORM_DECAY)
        hidden = alive & ~vis
        behind = n_beh[w] >= DORM_BEH * n_in[w]
        occ[w] = hidden & (cmax >= DORM_MINC) & (n_in[w] > 0) & behind
        sup[w] = hidden & (n_in[w] > 0) & ~behind & (c[w] < DORM_SUPC) & (S['birth'] < r)
        st['visible'] += int(vis.sum()); st['occluded'] += int(occ[w].sum())
        st['out_of_view'] += int((hidden & (n_in[w] == 0)).sum()); st['superfluous'] += int(sup[w].sum())
    S['cmax'] = cmax
    dk_old = S['dk'].clone()
    dk = torch.stack([occ[wj[j]] for j in range(K)], 1)                               # [N,K]
    dk[:, 0] = False                                                                 # knot 0 is always sent
    for k in TRACKED:
        v = P[k].data
        if k != 'means':
            for j in range(1, K):                                                     # hold the last non-dormant knot
                m = dk[:, j]
                v[m, j] = v[m, j - 1]
        so = opts[k].state.get(P[k], {})
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in so:
                so[key][dk] = 0
    S['dk'] = dk
    nb = torch.full((N, FOLLOW_K), -1, dtype=torch.long, device=dev)
    rows = torch.nonzero(dk.any(1)).squeeze(1)
    cand = torch.nonzero(~dk.any(1)).squeeze(1)
    if len(rows) and len(cand) >= FOLLOW_K:
        x0 = P['means'].data[:, 0]
        for i in range(0, len(rows), 512):
            rr = rows[i:i + 512]
            d = torch.cdist(x0[rr], x0[cand])
            nb[rr] = cand[d.topk(FOLLOW_K, largest=False).indices]
    S['nb'] = nb
    follow_means(P, S)
    st.update(dormant_splats=int(dk.any(1).sum()), dormant_knots=int(dk.sum()), woken_knots=int((dk_old & ~dk).sum()))
    return st, sup


@torch.no_grad()
def prune_superfluous(P, opts, S, g0, sup):
    """superfluous splats (dormant_update) die at the start of the first window that classifies them so."""
    death = S['death'].clone()
    n = 0
    for w in range(sup.shape[0]):
        r = g0 + w * REFRESH
        m = sup[w] & (death > r)
        death[m] = r; n += int(m.sum())
    if n == 0:
        return 0
    S['death'] = death
    keep = torch.nonzero(S['birth'] < death).squeeze(1)
    newidx = torch.full((len(death),), -1, dtype=torch.long, device=dev)
    newidx[keep] = torch.arange(len(keep), device=dev)
    rebuild(P, opts, S, keep, {}, None)
    if 'nb' in S:                                             # row indices changed: re-map the neighbour table
        S['nb'] = torch.where(S['nb'] >= 0, newidx[S['nb'].clamp_min(0)], S['nb'])
    return n


@torch.no_grad()
def opacity_constraints(P):
    """P13: floor on the opacity logits (never under the alpha cull) and/or tied opacity knots (OPA_K1)."""
    v = P['opacities'].data
    if OPA_FLOOR is not None:
        v.clamp_(min=OPA_FLOOR)
    if OPA_K1:
        v[:, 1:] = v[:, :1]


def error_select(P, S, gts, g0, Lg, K, ev_i):
    """P13 E7_ERRGROW: per refresh window the top ERR_N splats by excess-error mass (see the switch comment).
    Returns ([nW, N] bool selection, stats)."""
    from gsplat import rasterization
    nW = (Lg + REFRESH - 1) // REFRESH
    N = len(S['birth'])
    sel = torch.zeros(nW, N, dtype=torch.bool, device=dev)
    st = {'err_px': 0, 'hole_px': 0, 'sel': 0}
    mass = torch.zeros(nW, N, device=dev)                                           # per window and splat (P13 reloc)
    rng = np.random.default_rng(SEED * 1000 + g0 * 7 + ev_i)
    for w in range(nW):
        r = g0 + w * REFRESH
        t = min(r + (ev_i * 3) % REFRESH, g0 + Lg - 1)
        idx = alive_idx(S, t)
        cams = [int(c) for c in rng.choice(e3.train, min(ERR_CAMS, len(e3.train)), replace=False)]
        with torch.no_grad():
            q = curve_at(P, idx, t, g0, Lg, K)
            rgb, alpha, info = e3.render(q, cams)
            g = gt_now(gts[t - g0][cams])
            fg = (g.amax(-1) > 2 / 255).float()                                      # GT foreground [C,H,W]
            e = (rgb - g).abs().mean(-1) * fg
            if ERR_TILE > 1:
                e = Fn.avg_pool2d(e[:, None], ERR_TILE, stride=1, padding=ERR_TILE // 2,
                                  count_include_pad=False)[:, 0, :H, :W]
            ex = torch.relu(e - ERR_T) * fg
            if ERR_DETAIL:                  # owner: detail need = local error x GT frequency content (Laplace energy per tile)
                gy = g.mean(-1)[:, None]
                lap = Fn.conv2d(gy, torch.tensor([[[[0., 1, 0], [1, -4, 1], [0, 1, 0]]]], device=dev), padding=1).abs()
                Fd = Fn.avg_pool2d(lap, 8, stride=1, padding=4, count_include_pad=False)[:, 0, :H, :W]
                Fd = Fd / ((Fd * fg).sum() / fg.sum().clamp_min(1)).clamp_min(1e-8)
                ex = ex * Fd
            hole = ex * (alpha[..., 0] < 0.5).float()
        a = torch.ones((len(idx), 1), device=dev, requires_grad=True)
        img, _, inf2 = rasterization(q['means'].detach(), q['quats'].detach(), torch.exp(q['scales'].detach()),
                                     torch.sigmoid(q['opacities'].detach()), a, e3.VM[cams], e3.K[cams], W, H,
                                     sh_degree=None, packed=False)
        (img[..., 0] * ex).sum().backward()
        with torch.no_grad():
            m = a.grad[:, 0].clone()                                                 # sum_p e'(p) w_i(p)
            if ERR_HOLE and float(hole.sum()) > 0:
                rad = inf2['radii']
                vis = (rad > 0).all(-1) if rad.dim() == 3 else rad > 0               # [C,n]
                for ci in range(len(cams)):
                    hv, hu = torch.nonzero(hole[ci] > 0, as_tuple=True)
                    vi = torch.nonzero(vis[ci]).squeeze(1)
                    if len(hv) == 0 or len(vi) == 0:
                        continue
                    if len(hv) > 4096:                                               # subsample big holes
                        pk = torch.randperm(len(hv), device=dev)[:4096]
                        hv, hu = hv[pk], hu[pk]
                    hw = hole[ci][hv, hu] * (float((hole[ci] > 0).sum()) / len(hv))
                    pts = torch.stack([hu.float(), hv.float()], 1)
                    uv = inf2['means2d'][ci, vi].detach()
                    nn = torch.cat([torch.cdist(pts[i:i + 256], uv).argmin(1) for i in range(0, len(pts), 256)])
                    m.index_add_(0, vi[nn], hw)
                st['hole_px'] += int((hole > 0).sum())
            st['err_px'] += int((ex > 0).sum())
            mass[w, idx] = m
            cand = torch.nonzero(m >= ERR_MIN).squeeze(1)
            if len(cand) > ERR_N:
                cand = cand[torch.topk(m[cand], ERR_N).indices]
            sel[w, idx[cand]] = True
            st['sel'] += len(cand)
    return sel, st, mass


def detail_map(rgb, g, fg):
    """P17: per-pixel detail need d [C,H,W], mean 1 on the GT foreground: GT frequency content (|Laplacian| of the grey
    GT, tile mean over 8 px at 512 = 16 px at 1024) [x local L1 error, same tile, for E7_DETAIL=errfreq]."""
    tile = max(2, int(round(8 * W / 512)))
    lap = Fn.conv2d(g.mean(-1)[:, None], torch.tensor([[[[0., 1, 0], [1, -4, 1], [0, 1, 0]]]], device=dev), padding=1).abs()
    d = Fn.avg_pool2d(lap, tile, stride=1, padding=tile // 2, count_include_pad=False)[:, 0, :H, :W]
    if DETAIL == 'errfreq':
        e = (rgb - g).abs().mean(-1)[:, None]
        d = d * Fn.avg_pool2d(e, tile, stride=1, padding=tile // 2, count_include_pad=False)[:, 0, :H, :W]
    return d * fg / ((d * fg).sum() / fg.sum().clamp_min(1)).clamp_min(1e-12)


def detail_weights(P, S, gts, g0, Lg, K, ev_i):
    """P17 E7_DETAIL: [nW, N] per splat and refresh window D_i = sum_p w_i(p) d(p) / sum_p w_i(p) over DETAIL_CAMS
    train views of one frame of the window (w = rendering weight, d = detail_map); clamp [1/4, 4] ^ DETAIL_GAMMA; splats
    without foreground weight in these views get 1 (neutral)."""
    from gsplat import rasterization
    nW = (Lg + REFRESH - 1) // REFRESH
    N = len(S['birth'])
    num = torch.zeros(nW, N, device=dev); den = torch.zeros(nW, N, device=dev)
    rng = np.random.default_rng(SEED * 1000 + g0 * 7 + ev_i + 17)
    for w in range(nW):
        r = g0 + w * REFRESH
        t = min(r + (ev_i * 3) % REFRESH, g0 + Lg - 1)
        idx = alive_idx(S, t)
        cams = [int(c) for c in rng.choice(e3.train, min(DETAIL_CAMS, len(e3.train)), replace=False)]
        with torch.no_grad():
            q = curve_at(P, idx, t, g0, Lg, K)
            rgb, _, _ = e3.render(q, cams)
            g = gt_now(gts[t - g0][cams])
            fg = (g.amax(-1) > 2 / 255).float()
            d = detail_map(rgb, g, fg)
        a = torch.ones((len(idx), 2), device=dev, requires_grad=True)
        img, _, _ = rasterization(q['means'].detach(), q['quats'].detach(), torch.exp(q['scales'].detach()),
                                  torch.sigmoid(q['opacities'].detach()), a, e3.VM[cams], e3.K[cams], W, H,
                                  sh_degree=None, packed=False)
        (img[..., 0] * d + img[..., 1] * fg).sum().backward()
        num[w, idx] = a.grad[:, 0]; den[w, idx] = a.grad[:, 1]
    D = torch.where(den > 1e-3, num / den.clamp_min(1e-8), torch.ones_like(num))
    return D.clamp(0.25, 4.0) ** DETAIL_GAMMA


@torch.no_grad()
def relocate(P, opts, S, g0, Lg, K, mass=None):
    """P13 E7_RELOC (MCMC-style relocation, Kheradmand et al. 2024, adapted to per-section curves): a splat whose opacity
    knots are all below RELOC_T from knot j* to the end of the section (invisible in every frame from knot j* on, the
    "silent death") keeps its id and lifetime; its knots j*+1..K-1 (all knots if j* = 0) are replaced by those of a
    target splat drawn with probability ~ its mean opacity over those knots (RELOC_W=opa, MCMC) or ~ its excess-error
    mass (err) or the product (opaerr). Target and moved splat share the target's opacity like MCMC relocation
    (o' = 1 - (1 - o)^(1/(n+1)) for n splats moved onto one target); the moved copy gets a position offset drawn from
    the target's Gaussian (breaks the symmetry, MCMC adds SGLD noise instead) and fresh Adam state. Frames before knot j*
    are untouched; between knot j* and j*+1 the splat fades in while moving (one hat segment). shN: copied from the target
    only for splats born in this section (continuing splats' shN is already in the bitstream). Splat count is unchanged;
    in the codec this is a new curve for the same id (no birth/death record). Returns the number moved."""
    N = len(S['birth'])
    if N == 0:
        return 0
    op = torch.sigmoid(P['opacities'])                                              # [N,K]
    sfx = torch.flip(torch.cummax(torch.flip(op, [1]), 1).values, [1])              # max over knots j..K-1
    below = sfx < RELOC_T                                                           # [N,K] dead from knot j on
    jstar = torch.where(below.any(1), below.float().argmax(1), torch.full((N,), K, device=dev))
    knot_f = [g0 + j * span_of(Lg) / (K - 1) for j in range(K)]
    g1 = g0 + Lg
    full = S['death'] >= g1                                                         # lives to the section end
    cont = S['birth'] < g0
    moved = 0
    win_of = lambda f: min(int((f - g0) // REFRESH), (Lg + REFRESH - 1) // REFRESH - 1)
    for j in range(K - 1):                                                           # j* = K-1: nothing to move
        j1 = 0 if (j == 0 and SOFT0) else j + 1                                     # first replaced knot (geometry)
        jo = j1 if (j1 == 0 or not RELOC_HIDE) else j1 + 1                          # first knot with target opacity
        if jo > K - 1:
            continue
        dead = torch.nonzero((jstar == j) & full).squeeze(1)
        if len(dead) == 0:
            continue
        if RELOC_MAX > 0 and len(dead) > RELOC_MAX:
            dead = dead[torch.randperm(len(dead), device=dev)[:RELOC_MAX]]
        # targets: alive over the replaced knots (born at or before knot j1's frame, alive to the section end)
        ok = full & (S['birth'] <= knot_f[j1]) & (jstar > K - 1)
        w = op[:, jo:].mean(1) if RELOC_W in ('opa', 'opaerr') else torch.ones(N, device=dev)
        if RELOC_W in ('err', 'opaerr') and mass is not None:
            w = w * mass[win_of(knot_f[jo]):].sum(0)
        w = w * ok.float()
        if float(w.sum()) <= 0:
            continue
        tg = torch.multinomial(w, len(dead), replacement=True)
        cnt = torch.bincount(tg, minlength=N).float()                               # moved splats per target
        sl, so = slice(j1, K), slice(jo, K)
        o_t = op[tg, so]
        o_new = 1 - (1 - o_t) ** (1.0 / (cnt[tg, None] + 1))
        q = Fn.normalize(P['quats'][tg, sl], dim=-1)
        R = normalized_quat_to_rotmat(q.reshape(-1, 4)).reshape(len(tg), K - j1, 3, 3)
        s = torch.exp(P['scales'][tg, sl])
        z = torch.randn(len(tg), 1, 3, device=dev).expand(len(tg), K - j1, 3)
        off = torch.einsum('nkij,nkj,nkj->nki', R, s, z)
        for k in TRACKED:
            if k != 'opacities':
                P[k].data[dead, sl] = P[k].data[tg, sl]
        P['means'].data[dead, sl] += off
        if jo > j1:                     # RELOC_HIDE: knot j1 keeps the dead opacity, so the splat moves while invisible
            P['opacities'].data[dead, j1] = P['opacities'].data[dead, j1].clamp_max(float(np.log(RELOC_T / (1 - RELOC_T))))
        P['opacities'].data[dead, so] = torch.logit(o_new.clamp(1e-6, 1 - 1e-6))
        ut = torch.unique(tg)                                                       # targets: same split opacity
        o_ut = op[ut, so]
        P['opacities'].data[ut, so] = torch.logit((1 - (1 - o_ut) ** (1.0 / (cnt[ut, None] + 1))).clamp(1e-6, 1 - 1e-6))
        nb = dead[~cont[dead]]
        if len(nb):
            P['shN'].data[nb] = P['shN'].data[tg[~cont[dead]]]
        for k, o in opts.items():                                                   # fresh Adam state of the moved rows
            st = o.state.get(o.param_groups[0]['params'][0], {})
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in st:
                    if k in TRACKED:
                        st[key][dead, sl] = 0
                    elif len(nb):
                        st[key][nb] = 0
        jstar[dead] = K                                                              # now visible: not a target/dead again
        moved += len(dead)
    return moved
# ----------------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def csm_weights(P, S, gts, g0, Lg, K, S_kn, ext, chunk=6):
    """E7_CSMOOTH (P22): per-splat weight [N] (1 = the GT colour along the splat's track does not change) + stats dict.
    Renders every training camera at every frame of the section at full resolution (RGB+ED) with the current curves."""
    s_old = SCALE
    set_res(1.0)
    N = len(S['birth'])
    cams = list(e3.train)
    C = len(cams)
    s1 = torch.zeros(C, N, 3, device=dev); s2 = torch.zeros(C, N, 3, device=dev); cnt = torch.zeros(C, N, device=dev)
    path = torch.zeros(N, device=dev); last = torch.full((N, 3), float('nan'), device=dev)
    for t in range(g0, g0 + Lg):
        idx = alive_idx(S, t)
        if len(idx) == 0:
            continue
        q = cx.curve_adaptive(curve_at, P, idx, t, g0, Lg, S_kn) if S_kn is not None else curve_at(P, idx, t, g0, Lg, K)
        q = {k: v.detach() for k, v in q.items()}
        m = q['means']
        d = (m - last[idx]).norm(dim=-1)
        path.index_add_(0, idx, torch.nan_to_num(d, nan=0.0))
        last[idx] = m
        for a in range(0, C, chunk):
            cb = cams[a:a + chunk]
            rgb, alp, info, ed = render_train(q, cb, depth=True)
            g = gts[t - g0][cb].float() / 255.0                            # [c, H, W, 3] (full resolution)
            if CSM_BOX > 1:
                g = Fn.avg_pool2d(g.permute(0, 3, 1, 2), CSM_BOX, stride=1, padding=CSM_BOX // 2,
                                  count_include_pad=False).permute(0, 2, 3, 1)
            xy = info['means2d']                                           # [c, n, 2] pixel coords
            rad = info['radii']
            seen = (rad > 0).all(-1) if rad.dim() == 3 else rad > 0
            px = xy[..., 0].round().long(); py = xy[..., 1].round().long()
            seen = seen & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            pxc, pyc = px.clamp(0, W - 1), py.clamp(0, H - 1)
            ci = torch.arange(len(cb), device=dev)[:, None].expand_as(pxc)
            e_at = ed[ci, pyc, pxc]
            a_at = alp[..., 0][ci, pyc, pxc]
            vis = seen & (e_at > 0) & (a_at > 0.5) & (info['depths'] <= e_at * (1 + CSM_DTOL))
            col = g[ci, pyc, pxc]                                          # [c, n, 3]
            for j in range(len(cb)):
                v = vis[j]
                rows = idx[v]
                s1[a + j].index_add_(0, rows, col[j][v]); s2[a + j].index_add_(0, rows, col[j][v] ** 2)
                cnt[a + j].index_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
    ok = cnt >= CSM_MINF                                                   # [C, N]
    n_ = cnt.clamp_min(1)[..., None]
    std = ((s2 / n_ - (s1 / n_) ** 2).clamp_min(0)).sqrt().mean(-1)      # [C, N] per camera RMS deviation (mean over RGB)
    wc = cnt * ok
    r = (std * wc).sum(0) / wc.sum(0).clamp_min(1)
    has = ok.any(0)
    w = ((CSM_HI - r) / max(CSM_HI - CSM_LO, 1e-9)).clamp(0, 1)
    w = torch.where(has, w, torch.full_like(w, CSM_UNSEEN))
    moving = path > CSM_MOVE * 1e-3 * ext
    life = (S['birth'] < g0 + Lg) & (S['death'] > g0)
    real = has & (r >= CSM_HI); static_c = has & (r <= CSM_LO)
    st = {'n': int(life.sum()), 'seen': int((has & life).sum()), 'w_mean': float(w[life].mean()) if bool(life.any()) else 0.0,
          'w1': int(((w >= 0.999) & life).sum()), 'w0': int(((w <= 0.001) & life).sum()),
          'moving': int((moving & life).sum()),
          'moving_static_colour': int((moving & static_c & life).sum()), 'moving_real_change': int((moving & real & life).sum()),
          'still_static_colour': int((~moving & static_c & life).sum()), 'still_real_change': int((~moving & real & life).sum()),
          'r_quantiles': [float(x) for x in torch.quantile(r[has & life][:200000].float(),
                                                             torch.tensor([.1, .25, .5, .75, .9], device=dev)).tolist()]
          if bool((has & life).any()) else None}
    for k in CSM_ATTRS:                                                    # the model's own variation on the "no change" splats
        v = P[k].detach()
        if v.dim() == 2:
            v = v[..., None]
        rng = (v.amax(1) - v.amin(1)).mean(-1)
        sel = (w >= 0.999) & life
        st[f'model_range_{k}_w1'] = float(rng[sel].mean()) if bool(sel.any()) else None
        st[f'model_range_{k}_w0'] = float(rng[(w <= 0.001) & has & life].mean()) if bool(((w <= 0.001) & has & life).any()) else None
    set_res(s_old)
    return w, r, st


def csm_loss(P, S, steps):
    """E7_CSMOOTH term (P22): weighted L1 of every knot to the splat's knot mean (+ knot 0 to the previous end value)."""
    w = S['csm_w']
    N = max(len(w), 1)
    loss = 0.0
    for k in CSM_ATTRS:
        v = P[k]
        st = steps[k]
        if CSM_MODE == 'd2' and v.shape[1] > 2:                        # low-pass: weighted second differences (keeps drift)
            dv = (cx.curv_d2(v, v.shape[1], BASIS == 'bspline') if CURV_GREV else (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2])).abs() / st
        else:
            dv = (v - v.mean(1, keepdim=True)).abs() / st
        dv = dv.reshape(dv.shape[0], dv.shape[1], -1).mean(-1)            # [N, K or K-2]
        loss = loss + CSMOOTH * (w[:, None] * dv).sum() / (N * v.shape[1])
        if f'csm_prev_{k}' in S:
            has = S['csm_has'] * w
            d0 = ((v[:, 0] - S[f'csm_prev_{k}']).abs() / st).reshape(len(w), -1).mean(-1)
            loss = loss + CSMOOTH * (has * d0).sum() / N
    return loss


@torch.no_grad()
def static_pixels(gts):
    """P36: [C, H0, W0] bool, True where the GT of that camera does not change over the section (gts [Lg,C,H0,W0,3] u8)."""
    ref = gts[0].float()
    d = torch.zeros(gts.shape[1:4], device=dev)
    for t in range(1, gts.shape[0]):
        dt = (gts[t].float() - ref).abs().mean(-1)[:, None] / 255.0                 # [C,1,H,W]
        d = torch.maximum(d, Fn.avg_pool2d(dt, 5, stride=1, padding=2)[:, 0])        # 5x5 box against sensor noise
    mov = Fn.max_pool2d((d > STATIC_T).float()[:, None], STATIC_DIL, stride=1, padding=STATIC_DIL // 2)
    return mov[:, 0] < 0.5


@torch.no_grad()
def static_rows(P, smask):
    """P36: [N] bool - knot-0 centre projects (full-res intrinsics) into a static pixel in EVERY train camera whose image
    it lies in, and lies in at least one."""
    m = P['means'][:, 0]
    ok = torch.ones(m.shape[0], dtype=torch.bool, device=dev)
    seen = torch.zeros(m.shape[0], dtype=torch.int32, device=dev)
    Hm, Wm = smask.shape[1], smask.shape[2]
    for i in e3.train:
        c = e3.cams[i]
        R = torch.tensor(c['R'], device=dev, dtype=torch.float32); t = torch.tensor(c['t'], device=dev, dtype=torch.float32)
        Xc = m @ R.T + t
        z = Xc[:, 2]; zc = z.clamp_min(1e-6)
        u = (K0[i, 0, 0] * Xc[:, 0] / zc + K0[i, 0, 2]).long(); v = (K0[i, 1, 1] * Xc[:, 1] / zc + K0[i, 1, 2]).long()
        inimg = (z > 0) & (u >= 0) & (u < Wm) & (v >= 0) & (v < Hm)
        st = smask[i][v.clamp(0, Hm - 1), u.clamp(0, Wm - 1)]
        ok &= ~inimg | st
        seen += inimg.int()
    return ok & (seen > 0)


@torch.no_grad()
def probe0(P, S, gts, g0, Lg, K, it, tag, ev):
    """P36 E7_PROBE0: PSNR of frame g0 (test cams, mean over train cams) at this point of the section training."""
    idx = alive_idx(S, g0)
    q = curve_at(P, idx, g0, g0, Lg, K)
    rgb, _, _ = e3.render(q, e3.test)
    pt = e3.psnr(rgb, gt_now(gts[0][e3.test]))
    tr = [e3.psnr(e3.render(q, [c])[0], gt_now(gts[0][c:c + 1])) for c in e3.train]
    row = [int(it), tag, round(float(pt), 3), round(float(np.mean(tr)), 3), int(len(idx))]
    ev.setdefault('probe0', []).append(row)
    print(f'  [{g0}-{g0 + Lg}) probe0 {row}', flush=True)


def train_section(P, S, gts, g0, g1, K, ext):
    Lg = g1 - g0
    nW = (Lg + REFRESH - 1) // REFRESH
    lrs = lr_table(ext)
    opts = {k: torch.optim.Adam([P[k]], lr=lrs[k], eps=1e-15) for k in P}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts['means'], gamma=0.1 ** (1.0 / ITS))
    scheds_all = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=LRDECAY_ALL ** (1.0 / ITS))
                  for k, o in opts.items() if k != 'means'] if LRDECAY_ALL > 0 else []
    N = len(S['birth'])
    gsum = torch.zeros(nW, N, device=dev); gcnt = torch.zeros(nW, N, device=dev)
    cont = S['birth'] < g0                                        # continuing splats: knot 0 and shN frozen
    stop = 0 if FINETUNE else int(RSTOP * ITS)                    # finetune: no densify/prune (ids and lifetimes fixed)
    ts, ev = time.time(), {'cloned': 0, 'split': 0, 'pruned': 0}
    if DORM:
        S['dk'] = torch.zeros((len(S['birth']), K), dtype=torch.bool, device=dev)    # dormant knots of THIS section
        ev['dormant_events'] = []
    if OPA_FLOOR is not None or OPA_K1:
        opacity_constraints(P)
    fw = np.ones(Lg)
    if EDGE_W != 1 and Lg > 2 * EDGE_N:                           # the end knots carry half a hat each: sample more
        fw[:EDGE_N] = EDGE_W; fw[-EDGE_N:] = EDGE_W
    if MSAMP > 0 or TAIL_W != 1:                                  # P24: motion-weighted sampling + section-end weight
        mw, me = fgd.motion_weights(gts, e3.train, MSAMP, TAIL_W, TAIL_N)
        fw = fw * mw
        ev['msamp_w'] = [round(float(x), 4) for x in mw]
        if me is not None:
            ev['msamp_energy'] = [round(float(x), 6) for x in me]
    fw /= fw.sum()
    fsec, Kf, fl_acc = None, None, {'n': 0}
    if FLOW > 0 and Lg >= 2:                                      # P24: GT flow of this section (pinned CPU)
        tw = time.time()
        fsec = fgd.FlowSection(FLOW_DIR, g0, g1, len(e3.cams), dev)
        Kf = K0.clone(); Kf[:, :2, :] *= fsec.scale
        ev['flow_load_s'] = round(time.time() - tw, 1)
    # rough quantisation steps of codec/prep (qpos 0.001, R x127, 8 bit over typical ranges) for the curvature term
    curv_steps = {'means': 1e-3 * ext, 'quats': 1 / 127, 'scales': 8 / 255, 'opacities': 20 / 255, 'sh0': 4 / 255}
    csm_steps = dict(curv_steps)                                  # P22 E7_CSMOOTH (all attrs selectable)
    spat_steps = {k: v for k, v in curv_steps.items() if k in SPAT_ATTRS}
    sm_steps = {k: v for k, v in curv_steps.items() if k in SMOOTH_ATTRS}
    curv_steps = {k: v for k, v in curv_steps.items() if k in CURV_ATTRS}
    nn = None

    knot_frames = [g0 + j * span_of(Lg) / (K - 1) for j in range(K)]   # P13 curriculum: frame of each knot
    dep = load_depth(g0, g1) if DEPTH > 0 else None                  # P17 depth guidance (CPU, uint16 bit pattern)
    lo_its = int(LOWRES_FRAC * ITS) if LOWRES != 1 else 0            # P17 resolution plan
    set_res(LOWRES if lo_its > 0 else 1.0)
    assert not KADAPT or (not (DORM or RELOC or ERRGROW or CURRIC > 0 or OPA_K1 or FINETUNE) and KADAPT_AT >= RSTOP), \
        'E7_KADAPT works with plain densify only (no DORMANT/RELOC/ERRGROW/CURRIC/OPA_K1/FINETUNE) and E7_KADAPT_AT >= E7_RSTOP'
    K_eff = KLO if KADAPT else K                                   # P16: columns in use during the densify phase
    S_kn = None                                                    # P16: per-splat K after the upgrade event
    up_it = int(KADAPT_AT * ITS)
    bsp = BASIS == 'bspline'
    smooth_on = SMOOTH > 0 and 'sm_on' in S and bool((S['sm_on'] > 0).any())
    S.pop('csm_w', None)                                           # P22: weights are per section (recomputed below)
    smask = srows = None
    if STATIC_ON:                                                  # P36: static pixels of this section + static splats
        smask = static_pixels(gts)
        srows = static_rows(P, smask)
        ev['static_px_frac'] = round(float(smask[list(e3.train)].float().mean()), 4)
        ev['static_frac'] = [round(float(srows.float().mean()), 4)]
        print(f'  [{g0}-{g1}) P36 static: {ev["static_px_frac"]:.3f} of the train pixels, {ev["static_frac"][0]:.3f} of '
              f'the splats (lr x{STATIC_LR}, tie {STATIC_TIE})', flush=True)
    if PROBE0:
        probe0(P, S, gts, g0, Lg, K, 0, 'start', ev)
    for it in range(ITS):
        if it == lo_its and SCALE != 1.0:
            set_res(1.0)
        if MIXRES > 0 and it >= lo_its and LOWRES != 1:              # P17 multi-scale: keep the low scale in the mix
            want = LOWRES if np.random.rand() < MIXRES else 1.0
            if want != SCALE:
                set_res(want)
        if KADAPT and S_kn is None and it >= up_it:                # P16 adaptive K: upgrade the fastest splats to all K columns
            N_ = len(S['birth'])
            cx.sync_pad(P, TRACKED, torch.full((N_,), KLO, dtype=torch.long, device=dev))
            hi = cx.select_hi(P, S, KLO, KADAPT_FRAC, KADAPT_SCORE)
            with torch.no_grad():
                for k in TRACKED:
                    if bool(hi.any()):
                        P[k].data[hi] = cx.refit_columns(P[k].data[hi][:, :KLO], K, Lg, span_of(Lg), bsp)
            S_kn = torch.where(hi, torch.full_like(S['birth'], K), torch.full_like(S['birth'], KLO))
            S['kn'] = S_kn
            ev['k_hi'] = int(hi.sum()); ev['k_alive'] = int((S['death'] > g0).sum())
        loss, infos = 0.0, []
        n_act, j_free = Lg, K - 1
        if CURRIC > 0 and it < CURRIC * ITS:
            tau = g0 + (Lg - 1) * it / (CURRIC * ITS)                 # window end (fractional frame)
            n_act = min(Lg, max(2, int(tau - g0) + 2))                # frames g0 .. g0+n_act-1 are sampled
            j_free = min(K - 1, next((j for j in range(K) if knot_frames[j] > tau), K - 1))   # first knot past tau stays free
        pw = fw[:n_act] / fw[:n_act].sum()
        tsel = g0 + np.random.choice(n_act, FB, p=pw)
        cams_fix = None
        if TDIFF > 0 and n_act >= 2:                                  # P22: adjacent frame pair, same cameras
            t0_ = g0 + int(np.random.choice(n_act - 1, p=pw[:-1] / pw[:-1].sum()))
            tsel = np.array([t0_, t0_ + 1] + [t0_ + 1] * max(FB - 2, 0))[:max(FB, 2)]
            cams_fix = list(np.random.choice(e3.train, CB, replace=False))
        tpair = []
        for t in tsel:
            t = int(t)
            idx = alive_idx(S, t)
            q = cx.curve_adaptive(curve_at, P, idx, t, g0, Lg, S_kn) if S_kn is not None else curve_at(P, idx, t, g0, Lg, K_eff)
            cams = cams_fix if cams_fix is not None else list(np.random.choice(e3.train, CB, replace=False))
            if dep is not None:
                rgb, alp, info, ed = render_train(q, cams, depth=True)
            else:
                rgb, alp, info = render_train(q, cams)
            info['means2d'].retain_grad()
            gtv = gt_now(gts[t - g0][cams])
            loss = loss + e3.loss_fn(rgb, gtv)
            if cams_fix is not None and len(tpair) < 2:
                tpair.append((rgb, gtv))
            if len(infos) == 0:
                cams_first = cams
            if dep is not None:                 # P17: relative L1 of expected depth on GT-foreground pixels
                dg = depth_of(dep, t, g0, cams)
                m = (dg > 0).float()
                loss = loss + DEPTH * ((ed - dg).abs() / dg.clamp_min(1e-3) * m).sum() / m.sum().clamp_min(1)
            if BGALPHA > 0 or FGALPHA > 0:      # P13: alpha against the GT-derived matte (eroded bg -> 0, eroded fg -> 1)
                blk = (gtv.amax(-1) < 2 / 255).float()[:, None]
                bg = 1 - Fn.max_pool2d(1 - blk, 7, stride=1, padding=3)             # background eroded by 3 px
                fg = 1 - Fn.max_pool2d(blk, 7, stride=1, padding=3)                 # foreground eroded by 3 px
                a0 = alp[..., 0]
                loss = loss + BGALPHA * (a0 * bg[:, 0]).sum() / bg.sum().clamp_min(1) \
                    + FGALPHA * ((1 - a0) * fg[:, 0]).sum() / fg.sum().clamp_min(1)
            infos.append((t, idx, info))
        if len(tpair) == 2:                                           # P22 E7_TDIFF: match the temporal change of the GT
            (r0_, g0_), (r1_, g1_) = tpair
            loss = loss + TDIFF * FB * ((r1_ - r0_) - (g1_ - g0_)).abs().mean()
        loss = loss / FB
        if fsec is not None:                                          # P24 E7_FLOW: projected splat motion vs GT flow
            t_ = int(tsel[0])
            s_ = t_ if t_ + 1 < g1 else t_ - 1
            ia = alive_idx(S, s_)
            ia = ia[S['death'][ia] > s_ + 1]
            if len(ia):
                if S_kn is not None:
                    q0_ = cx.curve_adaptive(curve_at, P, ia, s_, g0, Lg, S_kn)
                    m1_ = cx.curve_adaptive(curve_at, P, ia, s_ + 1, g0, Lg, S_kn)['means']
                else:
                    q0_ = curve_at(P, ia, s_, g0, Lg, K_eff)
                    m1_ = curve_at(P, ia, s_ + 1, g0, Lg, K_eff)['means']
                Fg_, cf_ = fsec.get(s_, cams_first)
                lf_, fst_ = fgd.flow_term(q0_, m1_, e3.VM[cams_first], Kf[cams_first], fsec.w, fsec.h, Fg_, cf_,
                                          gts[s_ - g0][cams_first].float() / 255.0, FLOW_EDGE)
                loss = loss + FLOW * lf_
                fl_acc['n'] += 1
                for k_, v_ in fst_.items():
                    fl_acc[k_] = fl_acc.get(k_, 0.0) + v_
        if CURV > 0 and K_eff > 2:
            for k, st in curv_steps.items():
                v = P[k]
                if S_kn is None:
                    v = v[:, :K_eff]
                    d2 = cx.curv_d2(v, K_eff, bsp) if CURV_GREV else (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2])
                    loss = loss + CURV * (d2.abs() / st).mean()
                elif CURV_GREV:                                        # P16: per K group with its own Greville abscissae
                    num, cnt = 0.0, 0
                    for Kg in (KLO, K):
                        sel = S_kn == Kg
                        if bool(sel.any()) and Kg > 2:
                            d2 = cx.curv_d2(v[sel][:, :Kg], Kg, bsp).abs() / st
                            num = num + d2.sum(); cnt += d2.numel()
                    loss = loss + CURV * num / max(cnt, 1)
                else:                                                  # P16: only second differences inside each splat's own K
                    d2 = (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]).abs() / st
                    m = (torch.arange(K - 2, device=dev)[None] < (S_kn - 2)[:, None]).to(d2.dtype)
                    m = m.view(m.shape + (1,) * (d2.dim() - 2))
                    loss = loss + CURV * (d2 * m).sum() / (m.sum() * d2[0, 0].numel()).clamp_min(1.0)
        if smooth_on:                                                  # P16: section transition (continuing splats only)
            on = S['sm_on'] > 0
            kvec = S_kn if S_kn is not None else torch.full((len(S['birth']),), K_eff, dtype=torch.long, device=dev)
            for k, st in sm_steps.items():
                v0, sv = P[k][:, 0], cx.start_vel(P[k], kvec, span_of(Lg), bsp)
                pv = S[f'sm_vel_{k}']
                loss = loss + SMOOTH * (((v0 - (S[f'sm_val_{k}'] + pv)).abs() / st)[on].mean() + ((sv - pv).abs() / st)[on].mean())
        if SPAT > 0 and K > 1 and len(S['birth']) > SPAT_K:
            if nn is None or nn.shape[0] != len(S['birth']) or it % SPAT_EVERY == 0:
                nn = knn_indices(P['means'][:, 0].detach(), SPAT_K)
            loss = loss + SPAT * spat_loss(P, nn, spat_steps)
        if CSMOOTH > 0:                                                # P22: colour smoothing where the GT does not change
            if 'csm_w' not in S or (CSM_EVERY > 0 and it > 0 and it % CSM_EVERY == 0):
                tw = time.time()
                S['csm_w'], _, cst = csm_weights(P, S, gts, g0, Lg, K_eff, S_kn, ext)
                cst.update(it=it, secs=round(time.time() - tw, 1))
                ev.setdefault('csmooth', []).append(cst)
                print(f'  [{g0}-{g1}) it {it} csmooth weights {cst}', flush=True)
            loss = loss + csm_loss(P, S, csm_steps)
        if BOUND > 0 and 'csm_has' in S:                               # P22: value continuity at the section start
            hs = S['csm_has']
            for k in BOUND_ATTRS:
                d0 = ((P[k][:, 0] - S[f'csm_prev_{k}']).abs() / csm_steps[k]).reshape(len(hs), -1).mean(-1)
                loss = loss + BOUND * (hs * d0).sum() / hs.sum().clamp_min(1)

        if OPA_DECL > 0 and K > 2:                                    # P13: no fade-out of visible splats towards the section end
            a = torch.sigmoid(P['opacities'])                         # [N,K]
            vis = (a[:, :1].detach() > OPA_DECL_MIN).float()          # visible at the section start
            drop = torch.relu(a[:, :-1].detach() - a[:, 1:])          # decline between neighbouring knots (grad on the later one)
            loss = loss + OPA_DECL * (drop * vis).sum() / vis.sum().clamp_min(1.0) / (K - 1)
        loss.backward()
        loss = loss * FB
        if cont.any():
            if not SOFT0:
                for k in TRACKED:
                    P[k].grad[cont, 0] = 0
            if P['shN'].grad is not None:
                P['shN'].grad[cont] = 0
        if DORM and S['dk'].any():
            for k in TRACKED:
                P[k].grad[S['dk']] = 0
            if P['shN'].grad is not None:
                P['shN'].grad[S['dk'][:, 1:].all(1)] = 0
        refining = RS <= it < stop
        if refining:
            with torch.no_grad():
                for t, idx, info in infos:
                    w = (t - g0) // REFRESH
                    gr = (info['means2d'].absgrad if ABSGRAD else info['means2d'].grad).clone()   # [C,n,2]
                    gr[..., 0] *= W / 2.0 * len(gr) * FB
                    gr[..., 1] *= H / 2.0 * len(gr) * FB
                    rad = info['radii']
                    sel = (rad > 0).all(-1) if rad.dim() == 3 else rad > 0
                    _, gi = torch.where(sel)
                    rows = idx[gi]
                    gsum[w].index_add_(0, rows, gr[sel].norm(dim=-1))
                    gcnt[w].index_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
        st_old = None
        if srows is not None and bool(srows.any()):                   # P36: values of the static rows before the step
            st_old = {k: P[k].data[srows].clone() for k in list(TRACKED) + ['shN']}
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        sched.step()
        for sc_ in scheds_all:
            sc_.step()
        if st_old is not None:
            with torch.no_grad():
                for k, v0_ in st_old.items():
                    nv = v0_ + STATIC_LR * (P[k].data[srows] - v0_) if STATIC_LR != 1 else P[k].data[srows]
                    if STATIC_TIE and k in TRACKED:
                        nv = nv.mean(1, keepdim=True).expand_as(nv)
                    P[k].data[srows] = nv
        if CURRIC > 0 and j_free < K - 1:                             # P13 curriculum: later knots := first free knot
            with torch.no_grad():
                for k in TRACKED:
                    P[k].data[:, j_free + 1:] = P[k].data[:, j_free:j_free + 1]
        if OPA_FLOOR is not None or OPA_K1:
            opacity_constraints(P)
        if DORM and S['dk'].any():
            follow_means(P, S)
        reloc_ev = RELOC and it > RS and (it - RS) % RE == 0 and it < RELOC_STOP * ITS and not FINETUNE
        if (refining or reloc_ev) and it > RS and (it - RS) % RE == 0:
            extra = mass = None
            if (ERRGROW and refining) or (reloc_ev and RELOC_W != 'opa'):
                extra, est, mass = error_select(P, S, gts, g0, Lg, K, ev.setdefault('err_events', 0))
                ev['err_events'] += 1
                for kk, vv in est.items():
                    ev['err_' + kk] = ev.get('err_' + kk, 0) + vv
                if not ERRGROW:
                    extra = None
            if reloc_ev:
                ev['relocated'] = ev.get('relocated', 0) + relocate(P, opts, S, g0, Lg, K, mass)
        if refining and it > RS and (it - RS) % RE == 0:
            dw = None
            if DETAIL:
                dw = detail_weights(P, S, gts, g0, Lg, K_eff, ev.get('detail_events', 0))
                ev['detail_events'] = ev.get('detail_events', 0) + 1
                ev['detail_w_mean'] = float(dw.mean())
            if KADAPT:                                                # P16: densify sees a plain K_lo run (pad = last used column)
                cx.sync_pad(P, TRACKED, torch.full((len(S['birth']),), KLO, dtype=torch.long, device=dev))
            if PROBE0:
                probe0(P, S, gts, g0, Lg, K, it, 'pre', ev)
            c, s, p = refine(P, opts, S, gsum, gcnt, g0, Lg, K_eff, ext / 2, extra, dw)
            ev['cloned'] += c; ev['split'] += s; ev['pruned'] += p
            if PROBE0:
                probe0(P, S, gts, g0, Lg, K, it, 'post', ev)
            if DORM:
                cc, n_in, n_beh = contributions(P, S, g0, Lg, K)
                ds, sup = dormant_update(P, opts, S, g0, Lg, K, cc, n_in, n_beh)
                n_sup = prune_superfluous(P, opts, S, g0, sup)
                ds['pruned_superfluous'] = n_sup; ev['pruned'] += n_sup
                ev['dormant_events'].append({'it': it, **ds})
                if len(ev['dormant_events']) % 5 == 1:
                    print(f'  [{g0}-{g1}) it {it} dormant {ds}', flush=True)
            N = len(S['birth'])
            gsum = torch.zeros(nW, N, device=dev); gcnt = torch.zeros(nW, N, device=dev)
            cont = S['birth'] < g0
            if STATIC_ON:                                             # P36: rows changed - classify again
                srows = static_rows(P, smask)
                ev['static_frac'].append(round(float(srows.float().mean()), 4))
        if PROBE0 and ((it + 1) % PROBE0 == 0 or it == ITS - 1):
            probe0(P, S, gts, g0, Lg, K, it + 1, 'step', ev)
        if it % 1000 == 0 or it == ITS - 1:
            if fl_acc['n']:                                           # P24: mean flow residual since the last print
                fs_ = {k_: round(v_ / fl_acc['n'], 4) for k_, v_ in fl_acc.items() if k_ != 'n'}
                ev.setdefault('flow_log', []).append({'it': it, **fs_})
                print(f'  [{g0}-{g1}) it {it} flow {fs_}', flush=True)
                fl_acc = {'n': 0}
            print(f'  [{g0}-{g1}) it {it} N={len(S["birth"])} loss {float(loss) / FB:.4f} '
                  f'{(it + 1) / (time.time() - ts):.1f} it/s' + (f' res {W}' if W != W0 else ''), flush=True)
    set_res(1.0)
    if KADAPT:                                                        # P16: K per splat of this section; pad columns := last used one
        if S_kn is None:
            S['kn'] = torch.full_like(S['birth'], KLO)
            ev['k_hi'] = 0; ev['k_alive'] = int((S['death'] > g0).sum())
        cx.sync_pad(P, TRACKED, S['kn'])
    return ev


def finetune_source():
    """E7_FINETUNE: (curves.npz, tracks.npz, id -> fixed shN row [15,3] or None)."""
    cvf = np.load(os.path.join(FINETUNE, 'curves.npz'))
    trf = np.load(os.path.join(FINETUNE, 'tracks.npz'))
    fixed = None
    if SN_FIXED:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from codec.snvq import Assignment
        A = Assignment.load(SN_FIXED)
        fixed = {}
        for kidx in sorted(A.groups):                          # first key interval of an id wins (one static SN per splat)
            ids = A.groups[kidx]['ids'].tolist()
            rows = A.float_rows(kidx, range(len(ids)))
            for i, r in zip(ids, rows):
                fixed.setdefault(int(i), r.astype(np.float32).reshape(-1, 3))
    return cvf, trf, fixed


def finetune_section(cvf, trf, fixed, gi, g0, g1, K):
    """P, S of section gi from the source run (ids in codec order, lifetimes from tracks.npz like codec/prep)."""
    ids = cvf[f'g{gi}_ids']; b = trf['birth'][ids]; d = trf['death'][ids]
    sel = (b < g1) & (d > g0)
    rows = np.nonzero(sel)[0][np.argsort(ids[sel], kind='stable')]
    ids, b, d = ids[rows], b[rows], d[rows]
    P = {k: torch.tensor(np.ascontiguousarray(cvf[f'g{gi}_{k}'][rows]), dtype=torch.float32, device=dev) for k in TRACKED}
    assert P['means'].shape[1] == K
    shn = trf['shN'][ids].astype(np.float32)
    if fixed is not None:
        shn = np.stack([fixed.get(int(i), shn[j]) for j, i in enumerate(ids.tolist())])
    P['shN'] = torch.tensor(shn, device=dev)
    P = torch.nn.ParameterDict({k: torch.nn.Parameter(v, requires_grad=not (k == 'shN' and fixed is not None))
                                for k, v in P.items()})
    S = {'birth': torch.tensor(b.astype(np.int64), device=dev), 'death': torch.tensor(d.astype(np.int64), device=dev),
         'ids': torch.tensor(ids.astype(np.int64), device=dev)}
    return P, S


def run_k(K, init, ext):
    out_dir = os.path.join(RUN, f'k{K}')
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()
    cvf = trf = fixed = None
    if FINETUNE:
        cvf, trf, fixed = finetune_source()
        K = int(cvf['K'])
        out_dir = os.path.join(RUN, f'k{K}'); os.makedirs(out_dir, exist_ok=True)
        n0 = 0
        P = torch.nn.ParameterDict()
        S = {'birth': torch.zeros(0, dtype=torch.long, device=dev), 'death': torch.zeros(0, dtype=torch.long, device=dev),
             'ids': torch.zeros(0, dtype=torch.long, device=dev)}
        next_id = int(trf['birth'].shape[0])
    else:
        n0 = init['means'].shape[0]
        P = torch.nn.ParameterDict({k: torch.nn.Parameter(
            (v[:, None].expand((n0, K) + v.shape[1:]) if k in TRACKED and k != 'sh0' else
             v.expand((n0, K, 3)) if k == 'sh0' else v).clone()) for k, v in init.items()})   # sh0 [N,1,3] -> [N,K,3]
        S = {'birth': torch.zeros(n0, dtype=torch.long, device=dev), 'death': torch.full((n0,), BIG, dtype=torch.long, device=dev),
             'ids': torch.full((n0,), -1, dtype=torch.long, device=dev)}
        next_id = 0
    if DORM:
        S['cmax'] = torch.zeros(n0, device=dev); S['dage'] = torch.zeros(n0, dtype=torch.long, device=dev)
    birth_of, death_of, shN_of = {}, {}, {}
    csm_end, csm_save = None, {}                                   # P22: (ids, {attr: end value}) of the last section
    frames, curves = [], {}
    log = {'K': K, 'L': L, 'knots': KNOTS, 'its': ITS, 'refresh': REFRESH, 'fb': FB, 'cb': CB, 'grow': GROW, 'prune': PRUNE,
           're': RE, 'rs': RS, 'rstop': RSTOP, 'soft_knot0': SOFT0, 'edge_w': EDGE_W, 'edge_n': EDGE_N, 'curv': CURV, 'curv_attrs': CURV_ATTRS, 'spat': SPAT, 'spat_k': SPAT_K, 'spat_attrs': SPAT_ATTRS, 'ext': ext, 'T': T, 'gt': e3.GT,
           'finetune': FINETUNE or None, 'sn_fixed': SN_FIXED or None,
           'basis': BASIS, 'seed': SEED, 'tracks': TRACKS,
           'kadapt': {'klo': KLO, 'frac': KADAPT_FRAC, 'at': KADAPT_AT, 'score': KADAPT_SCORE} if KADAPT else None,
           'smooth': {'w': SMOOTH, 'attrs': SMOOTH_ATTRS} if SMOOTH > 0 else None,
           'csmooth': {'w': CSMOOTH, 'attrs': CSM_ATTRS, 'lo': CSM_LO, 'hi': CSM_HI, 'box': CSM_BOX, 'dtol': CSM_DTOL,
                       'minf': CSM_MINF, 'unseen': CSM_UNSEEN, 'every': CSM_EVERY, 'mode': CSM_MODE} if CSMOOTH > 0 else None,
           'tdiff': TDIFF,
           'p24': {'flow': FLOW, 'flow_dir': FLOW_DIR, 'flow_edge': FLOW_EDGE, 'msamp': MSAMP, 'tail_w': TAIL_W,
                   'tail_n': TAIL_N} if (FLOW > 0 or MSAMP > 0 or TAIL_W != 1) else None,
           'opa_lr': OPA_LR, 'opa_floor': OPA_FLOOR, 'opa_k1': OPA_K1, 'opa_decl': OPA_DECL, 'curric': CURRIC, 'dormant': DORM,
           'errgrow': {'t': ERR_T, 'n': ERR_N, 'min': ERR_MIN, 'cams': ERR_CAMS, 'tile': ERR_TILE, 'hole': ERR_HOLE,
                       'opa': ERR_OPA} if ERRGROW else None, 'absgrad': ABSGRAD,
           'reloc': {'t': RELOC_T, 'w': RELOC_W, 'max': RELOC_MAX, 'stop': RELOC_STOP, 'hide': RELOC_HIDE} if RELOC else None,
           'err_detail': ERR_DETAIL, 'bgalpha': BGALPHA, 'fgalpha': FGALPHA,
           'p17': {'cams': CAMSEL, 'n_train': len(e3.train), 'res': [W0, H0], 'lowres': LOWRES, 'lowres_frac': LOWRES_FRAC,
                   'init_scale': INIT_SCALE, 'cap': CAP, 'detail': DETAIL or None, 'detail_cams': DETAIL_CAMS,
                   'detail_gamma': DETAIL_GAMMA, 'depth': DEPTH, 'mixres': MIXRES},
           'dorm': {'frac': DORM_FRAC, 'minc': DORM_MINC, 'beh': DORM_BEH, 'tol': DORM_TOL, 'supc': DORM_SUPC, 'max': DORM_MAX,
                    'camstep': DORM_CAMSTEP, 'decay': DORM_DECAY, 'follow_k': FOLLOW_K} if DORM else None,
           'psnr': [], 'psnr_fg': [], 'alive': [], 'born': [], 'died': [], 'sections': []}
    for gi, g0 in enumerate(range(0, T, L)):
        g1 = min(T, g0 + L)
        Lg = g1 - g0
        ts = time.time()
        if FINETUNE:
            P, S = finetune_section(cvf, trf, fixed, gi, g0, g1, K)
        elif g0 > 0:                                            # warm start: drop the dead, all knots := last knot
            keep = torch.nonzero(S['death'] > g0).squeeze(1)
            sm = {}
            if SMOOTH > 0:                                      # P16: end value / velocity of the section that just finished
                kn_old = S['kn'][keep] if KADAPT else torch.full((len(keep),), K, dtype=torch.long, device=dev)
                for k in SMOOTH_ATTRS:
                    v_, w_ = cx.end_state(P[k][keep].detach(), kn_old, span_of(L), BASIS == 'bspline')
                    sm[f'sm_val_{k}'], sm[f'sm_vel_{k}'] = v_, w_
                sm['sm_on'] = torch.ones(len(keep), device=dev)
            P = torch.nn.ParameterDict({k: torch.nn.Parameter(
                (P[k][keep][:, -1:].expand_as(P[k][keep]) if k in TRACKED else P[k][keep]).detach().clone())
                for k in P})
            S = {k: v[keep] for k, v in S.items() if not (k == 'kn' or k.startswith('sm_'))}
            S.update(sm)
        gts = load_section(g0, g1)
        if CSMOOTH > 0 or BOUND > 0:                            # P22: previous section's end value per continuing id
            for k in [k for k in S if k.startswith('csm_')]:
                S.pop(k)
            if csm_end is not None and len(S['birth']):
                pid, pv = csm_end
                pos = torch.full((int(max(int(pid.max()), int(S['ids'].max())) + 2),), -1, dtype=torch.long, device=dev)
                pos[pid] = torch.arange(len(pid), device=dev)
                j = pos[S['ids'].clamp_min(0)]
                hs = (S['ids'] >= 0) & (j >= 0) & (S['birth'] < g0)
                S['csm_has'] = hs.float()
                for k in PREV_ATTRS:
                    S[f'csm_prev_{k}'] = torch.where(hs.view((-1,) + (1,) * (pv[k].dim() - 1)), pv[k][j.clamp_min(0)],
                                                     torch.zeros_like(pv[k][:1]).expand((len(j),) + pv[k].shape[1:]))
        its_keep = ITS
        if ITS0 > 0 and gi == 0:                                # P36 E7_ITS0: longer first section
            globals()['ITS'] = ITS0
        ev = train_section(P, S, gts, g0, g1, K, ext)
        if ITS0 > 0 and gi == 0:
            globals()['ITS'] = its_keep
            ev['its'] = ITS0
        if DORM:                                                # whole-section dormancy counter; die after DORM_MAX
            full = S['dk'][:, 1:].all(1)
            S['dage'] = torch.where(full, S['dage'] + 1, torch.zeros_like(S['dage']))
            kill = (S['dage'] > DORM_MAX) & (S['death'] > g1) & (DORM_MAX > 0)
            S['death'] = torch.where(kill, torch.full_like(S['death'], g1), S['death'])
            ev['dormant_end'] = {'dormant_splats': int(S['dk'].any(1).sum()), 'fully_dormant': int(full.sum()),
                                 'killed_long_dormant': int(kill.sum())}
            ev['dormant_last'] = ev['dormant_events'][-1] if ev['dormant_events'] else {}
            ev.pop('dormant_events')
        secs_train = time.time() - ts
        with torch.no_grad():
            new = S['ids'] < 0
            S['ids'][new] = torch.arange(next_id, next_id + int(new.sum()), device=dev)
            next_id += int(new.sum())
            ids_np, b_np, d_np = (S[k].cpu().numpy() for k in ('ids', 'birth', 'death'))
            shn_np = P['shN'].detach().cpu().numpy()
            for i, b, d, sn in zip(ids_np.tolist(), b_np.tolist(), d_np.tolist(), shn_np):
                birth_of[i] = b; death_of[i] = min(d, T); shN_of[i] = sn
            curves[f'g{gi}_ids'] = ids_np.astype(np.int32)
            curves[f'g{gi}_birth'] = b_np.astype(np.int32); curves[f'g{gi}_death'] = np.minimum(d_np, T).astype(np.int32)
            for k in TRACKED:
                curves[f'g{gi}_{k}'] = P[k].detach().cpu().numpy().astype(np.float32)
            if CSMOOTH > 0 or BOUND > 0:                        # P22: end values for the next section + weights (csm.npz)
                csm_end = (S['ids'].clone(), {k: P[k][:, -1].detach().clone() for k in PREV_ATTRS})
                if 'csm_w' in S:
                    csm_save[f'g{gi}_ids'] = ids_np.astype(np.int32)
                    csm_save[f'g{gi}_w'] = S['csm_w'].cpu().numpy().astype(np.float16)
            if DORM:                                            # dormant knots -> csgs v2 follow masks (codec/prep)
                curves[f'g{gi}_dk'] = S['dk'].cpu().numpy()
            if KADAPT:                                          # P16: trained K per splat (columns beyond it = copies of the last)
                curves[f'g{gi}_kn'] = S['kn'].cpu().numpy().astype(np.int16)
            for t in range(g0, g1):
                idx = alive_idx(S, t)
                q = cx.curve_adaptive(curve_at, P, idx, t, g0, Lg, S['kn']) if KADAPT else curve_at(P, idx, t, g0, Lg, K)
                rgb, _, _ = e3.render(q, e3.test)
                g = gts[t - g0][e3.test].float() / 255.0
                full, fg = e3.psnr(rgb, g), e3.psnr_fg(rgb, g)
                if TRACKS == 'thin':
                    m = q['means'].detach()
                    frames.append((None, {'means': torch.stack([m.amin(0), m.amax(0)]).cpu().numpy()}))
                else:
                    frames.append((S['ids'][idx].cpu().numpy(), {k: q[k].cpu().numpy() for k in TRACKED}))
                log['psnr'].append(full); log['psnr_fg'].append(fg); log['alive'].append(len(idx))
                log['born'].append(int(((S['birth'] == t) & (S['death'] > t)).sum()) if t > 0 else len(idx))
                log['died'].append(int((S['death'] == t).sum()) if t > 0 else 0)
                print(f'frame {t}: N={len(idx)} PSNR {full:.2f} dB, fg {fg:.2f} dB', flush=True)
                if t == 0 or t == T - 1:
                    e3.preview(gts[t - g0].float() / 255.0, rgb, os.path.join(out_dir, f'preview_f{t:03d}.png'))
        sec = {'g0': g0, 'g1': g1, 'rows': len(S['birth']), 'secs_train': round(secs_train, 1),
               'secs': round(time.time() - ts, 1), **ev, 'mean_psnr_fg': float(np.mean(log['psnr_fg'][g0:g1]))}
        log['sections'].append(sec); print('section', sec, flush=True)
        del gts
    # assemble over the global id space (same layout as e3 v3)
    Ng = next_id
    if TRACKS == 'thin':
        out = {'means': np.stack([a['means'] for _, a in frames]).astype(np.float32), 'thin': True}     # [T, 2, 3] envelope
    else:
        out = {k: np.full((T, Ng) + frames[0][1][k].shape[1:], np.nan, np.float16 if k in ('scales', 'opacities', 'sh0') else np.float32)
               for k in TRACKED}
        for t, (fid, attrs) in enumerate(frames):
            for k in TRACKED:
                out[k][t, fid] = attrs[k]
    tb = np.asarray(trf['birth']) if trf is not None else None    # once: NpzFile re-reads a member on every access
    td = np.asarray(trf['death']) if trf is not None else None
    b = np.array([birth_of.get(i, int(tb[i]) if tb is not None else 0) for i in range(Ng)], np.int32)
    d = np.array([death_of.get(i, int(td[i]) if td is not None else 0) for i in range(Ng)], np.int32)
    sn = np.zeros((Ng,) + next(iter(shN_of.values())).shape, np.float16)
    if trf is not None:
        sn[:] = trf['shN'][:Ng]
    for i, v in shN_of.items():
        sn[i] = v
    np.savez(os.path.join(out_dir, 'tracks.npz'), birth=b, death=d, shN=sn, **out)
    np.savez(os.path.join(out_dir, 'curves.npz'), L=L, K=K, knots=KNOTS, soft_knot0=SOFT0, basis=BASIS, **curves)
    if csm_save:
        np.savez(os.path.join(out_dir, 'csm.npz'), **csm_save)
    fg = np.array(log['psnr_fg'])
    log.update(N_total=Ng, mean_psnr=float(np.mean(log['psnr'])), mean_psnr_fg=float(fg.mean()), min_psnr_fg=float(fg.min()),
               last_psnr_fg=float(fg[-1]), alive_mean=float(np.mean(log['alive'])), secs_total=time.time() - t0,
               secs_init=init_secs)
    json.dump(log, open(os.path.join(out_dir, 'e7.json'), 'w'), indent=1)
    print(f'E7 K={K} OK {out_dir}: ids {Ng}, alive/frame {log["alive_mean"]:.0f}, mean PSNR {log["mean_psnr"]:.2f} dB, '
          f'fg mean {log["mean_psnr_fg"]:.2f} / min {log["min_psnr_fg"]:.2f} / last {log["last_psnr_fg"]:.2f} dB, '
          f'{log["secs_total"]:.0f} s', flush=True)


init_secs = 0.0


def main():
    global init_secs
    os.makedirs(RUN, exist_ok=True)
    path = os.path.join(RUN, 'init.pt')
    if FINETUNE:                                               # S1: no frame-0 fit, only the scene extent of the source run
        src = json.load(open(os.path.join(FINETUNE, 'e7.json')))
        init, ext = None, float(src['ext'])
        print(f'finetune from {FINETUNE} (ext {ext:.3f}, fixed SN: {SN_FIXED or "no"})', flush=True)
    elif os.path.exists(path):                                 # frame-0 fit shared by every K of this RUN
        d = torch.load(path, map_location=dev)
        init, ext, init_secs = d['p'], d['ext'], d.get('secs', 0.0)
        print(f'init from {path}: N={init["means"].shape[0]}', flush=True)
    else:
        t0 = time.time()
        torch.manual_seed(SEED); np.random.seed(SEED)
        gt0 = e3.load_frame(0)
        p, ext = e3.init_points(gt0)                             # (colours indexed at full-res depth pixels)
        if INIT_SCALE != 1.0:                                    # P17: frame-0 fit at a lower render scale
            set_res(INIT_SCALE); gt0 = gt_now(gt0)
        p, _ = e3.fit_densify(p, torch.arange(p['means'].shape[0], device=dev), gt0, e3.IT0, ext, first=True)
        if CAP > 0 and p['means'].shape[0] > CAP:               # P17: the frame-0 pool obeys the cap too
            keep = torch.topk(p['opacities'].reshape(-1), CAP).indices
            print(f'init cap: {p["means"].shape[0]} -> {CAP} rows (highest opacity)', flush=True)
            p = {k: v[keep].contiguous() for k, v in p.items()}
        init_secs = time.time() - t0
        init = {k: v.detach() for k, v in p.items()}
        torch.save({'p': init, 'ext': ext, 'secs': init_secs}, path)
        rgb, _, _ = e3.render(init, e3.test)
        print(f'init fit: N={init["means"].shape[0]} fg {e3.psnr_fg(rgb, gt0[e3.test]):.2f} dB ({init_secs:.0f} s)', flush=True)
        set_res(1.0)
        del gt0
    for K in KS:
        run_k(K, init, ext)
        torch.cuda.empty_cache()
    summ = {}
    for f in sorted(glob.glob(os.path.join(RUN, 'k*', 'e7.json'))):
        j = json.load(open(f))
        summ[f'K{j["K"]}'] = {k: j[k] for k in ('K', 'mean_psnr_fg', 'min_psnr_fg', 'last_psnr_fg', 'mean_psnr', 'N_total',
                                                'alive_mean', 'secs_total', 'psnr_fg')}
    json.dump({'run': RUN, 'gt': e3.GT, 'T': T, 'L': L, 'results': summ}, open(os.path.join(RUN, 'e7.json'), 'w'), indent=1)
    print('E7 OK', RUN)


if __name__ == '__main__':
    main()
