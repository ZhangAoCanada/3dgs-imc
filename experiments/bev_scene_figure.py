#!/usr/bin/env python3
"""BEV floater figure WITH a scene-explainer header.

Diagnosis that drove this design:
  * A straight top-down RENDER of the scene is a novel-view extrapolation (no training
    camera is overhead) -> blurry + shows the floaters themselves. Not a clean background.
  * A colored top-down SfM projection is a muddy abstract blob (foliage reads blue/gray,
    no recognizable structure). The bonsai is recognizable only from the SIDE.
  => Recognition MUST come from a PERSPECTIVE photo; the top-down can only show camera/gap
     GEOMETRY. The 8 prior attempts failed on LINKING the perspective photo to the BEV
     (floating insets, scale mismatch, clutter) -- not on the components.

This figure:
  Header (scene explainer):
    [left]  a recognizable PERSPECTIVE GT view of the bonsai scene ("what is the scene").
    [right] a clean top-down SCHEMATIC in the exact floater crop frame: faint SfM footprint
            + blue train-camera dots + orange held-out gap wedge + orange gap-camera dots,
            with an arrow "floaters cluster in the held-out gap" pointing into the wedge.
            ("where is the gap, and where do floaters appear" -- in the BEV frame).
  Body: 9 methods x 5 gap angles free-space floater density (inferno over faint footprint),
        with a small orange wedge icon under each column header whose width = that angle,
        so the reader maps "orange wedge (header) = gap = where floaters appear".
The orange wedge is the single consistent link across photo-explainer, schematic, and grid.
"""
import os, sys, glob, json, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from PIL import Image
from plyfile import PlyData
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_crop import proj_from_ref, main_box_proj
import bev_map_context_v4_figure as b4

GA = "/data1/zhangao/outputs/viz_multiangle/global_all"
SCENE = sys.argv[1] if len(sys.argv) > 1 else "bonsai"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{GA}/BEV_scene_{SCENE}.png"
ORDER = ["3DGS", "Hier", "ScaffoldGS", "MCMC", "CDGS", "PGSR", "MH-3DGS", "MVGS", "Ours"]
NICE = {"3DGS":"3DGS","Hier":"Hier-GS","ScaffoldGS":"Scaffold-GS","MCMC":"3DGS-MCMC",
        "CDGS":"CDGS","PGSR":"PGSR","MH-3DGS":"MH-3DGS","MVGS":"MVGS","Ours":"Ours (GS-LSS)"}
BINS = 320
INTERP = "bicubic"

# ---- floater grid data (same construction as bev_grid_overlay) ----
refp = sorted(glob.glob(f"/data1/zhangao/outputs/baselines_multi_angle/{SCENE}/*/input.ply"))
ref = np.stack([PlyData.read(refp[0])["vertex"][k] for k in ("x","y","z")],1).astype(np.float32)
center, axes = proj_from_ref(ref)
r2 = (ref-center)@axes.T
pad=0.35; mn=r2.min(0)-pad*(r2.max(0)-r2.min(0)); mx=r2.max(0)+pad*(r2.max(0)-r2.min(0))
refH,_,_=np.histogram2d(r2[:,0],r2[:,1],bins=BINS,range=[[mn[0],mx[0]],[mn[1],mx[1]]])
refH=(refH.T>0).astype(float)
cx,cy,half=main_box_proj(ref,center,axes,mass_frac=0.88,margin=2.8)
i0,i1,j0,j1=b4.px_box(mn,mx,BINS,cx,cy,half) if hasattr(b4,'px_box') else (None,None,None,None)
# px_box lives in main_crop
from main_crop import px_box
i0,i1,j0,j1=px_box(mn,mx,BINS,cx,cy,half)
EXT=[cx-half,cx+half,cy-half,cy+half]
refS=refH[i0:i1,j0:j1]
ANGLES=[a for a in [20,40,60,80,100] if os.path.exists(f"{GA}/{SCENE}__Ours__{a}.npy")]
def frac(m,a):
    p=f"{GA}/{SCENE}__{m}.frac.json"
    if os.path.exists(p):
        d=json.load(open(p)); return d.get(str(a)) or d.get(str(float(a)))
    return None
grids={}
for m in ORDER:
    for a in ANGLES:
        p=f"{GA}/{SCENE}__{m}__{a}.npy"
        if not os.path.exists(p): grids[(m,a)]=None; continue
        full=np.load(p); fr=frac(m,a); s=float(full.sum())
        tm=(s/fr) if (fr and s>0) else None
        gn=(full/tm) if tm else full*0.0
        grids[(m,a)]=gn[i0:i1,j0:j1]
allv=[g for g in grids.values() if g is not None and g.size]
vmax=np.percentile(np.concatenate([g.ravel() for g in allv]),99.5)+1e-9

# ---- camera/gap geometry (top-down schematic) ----
center_g,axes_g,crop_ext,ref_path=b4.crop_extent(SCENE)
geoms=b4.build_geoms(SCENE,center_g,axes_g,crop_ext)
G100=geoms[100]
cc=G100['crop_center']; gap_dir=G100['gap_dir']; gap_half=G100['gap_half']
xy=G100['xy']; is_gap=G100['is_gap']; theta=G100['theta']
# wedge radius: distance from crop center to farthest camera
radius=float(np.linalg.norm(xy-cc[None,:],axis=1).max())*1.02

# ---- perspective photo (recognizable scene anchor) ----
# a known-good GT view (matches the prior context attempt's reference view)
photo_path="/data1/zhangao/outputs/viz_multiangle/cells/bonsai_20/renders/GT/00005.png"
if not os.path.exists(photo_path):
    for a in [20,60,100,40,80]:
        cands=sorted(glob.glob(f"/data1/zhangao/outputs/viz_multiangle/cells/{SCENE}_{a}/renders/GT/*.png"))
        if cands: photo_path=cands[0]; break
# crop to center the bonsai (removes the visually-dominant red bicycle on the left edge)
_im=Image.open(photo_path).convert("RGB"); _w,_h=_im.size
_box=(180,90,610,510) if (_w,_h)==(780,520) else (0,0,_w,_h)
photo=np.asarray(_im.crop(_box))/255.0

# ============================ FIGURE ============================
fig=plt.figure(figsize=(15,12.6),dpi=150)
gs=GridSpec(2,1,height_ratios=[1.0,2.7],hspace=0.16,figure=fig)
# ---- header ----
hg=GridSpecFromSubplotSpec=gs[0].subgridspec(1,2,width_ratios=[0.82,1.18],wspace=0.10)
axp=fig.add_subplot(hg[0,0]); axs=fig.add_subplot(hg[0,1])
# photo
axp.imshow(photo,aspect="equal",interpolation="bicubic")
axp.set_xticks([]); axp.set_yticks([])
axp.set_title("Bonsai scene  (a perspective view: a bonsai tree on a table in a room)",
              fontsize=12,fontweight="bold")
for s in axp.spines.values(): s.set_linewidth(1.0); s.set_edgecolor("#444")
# schematic
axs.set_facecolor("#fbfaf7")
axs.imshow(np.clip(refS,0,1),origin="lower",extent=EXT,cmap="Greys",vmin=0,vmax=1.2,
           alpha=0.22,aspect="equal",interpolation=INTERP)
# ONE clear 100deg held-out gap wedge (the widest, most dramatic). The body's dashed outlines
# (one per column, widening 20->100) carry the "5 angles" story; this schematic shows what the
# gap IS. Plus 2 thin outline guides at 20deg and 60deg so the reader sees the widening direction.
_t1=math.degrees(gap_dir-gap_half); _t2=math.degrees(gap_dir+gap_half)
axs.add_patch(Wedge(tuple(cc),radius,_t1,_t2,facecolor="#ee7b16",edgecolor="#ee7b16",lw=1.6,alpha=0.16,zorder=1))
for _a in (ANGLES[0], ANGLES[len(ANGLES)//2]):
    _h=math.radians(_a/2.0)
    axs.add_patch(Wedge(tuple(cc),radius,math.degrees(gap_dir-_h),math.degrees(gap_dir+_h),
        facecolor="none",edgecolor="#ee7b16",lw=1.0,linestyle=":",alpha=0.55,zorder=2))
# label the 100deg wedge
_lab_r=radius*1.04
axs.text(cc[0]+math.cos(gap_dir)*_lab_r, cc[1]+math.sin(gap_dir)*_lab_r,
         "100° held-out gap",color="#a14a00",fontsize=11,fontweight="bold",ha="center",va="center",zorder=12,
         bbox=dict(facecolor="white",edgecolor="none",alpha=0.9,pad=2.5))
# cameras
tr=~is_gap; gp=is_gap
axs.scatter(xy[tr,0],xy[tr,1],s=7,c="#1764d8",alpha=0.45,linewidths=0,zorder=3)
axs.scatter(xy[gp,0],xy[gp,1],s=16,c="#ee7b16",alpha=0.92,edgecolors="#a14a00",linewidths=0.4,zorder=5)
# arrow into the wedge: "floaters cluster here" (points along the gap center direction)
mid=gap_dir
arr_from=cc+np.array([math.cos(mid),math.sin(mid)])*radius*1.18
arr_to  =cc+np.array([math.cos(mid),math.sin(mid)])*radius*0.55
axs.add_patch(FancyArrowPatch(arr_from,arr_to,arrowstyle="-|>",mutation_scale=22,
               lw=2.0,color="#b00020",zorder=8))
axs.text(arr_from[0],arr_from[1]+radius*0.12,"floaters cluster\nin the held-out gap",
         ha="center",va="bottom",fontsize=11,fontweight="bold",color="#b00020",zorder=9,
         bbox=dict(facecolor="white",edgecolor="none",alpha=0.9,pad=2.5))
axs.set_xlim(EXT[0],EXT[1]); axs.set_ylim(EXT[2],EXT[3]); axs.set_aspect("equal")
axs.set_xticks([]); axs.set_yticks([])
axs.set_title("Top-down camera coverage  (orange = 100deg held-out gap; blue = training views)",
              fontsize=12,fontweight="bold")
for s in axs.spines.values(): s.set_linewidth(1.0); s.set_edgecolor("#444")
# schematic legend
axs.scatter([EXT[0]+0.06*(EXT[1]-EXT[0])],[EXT[3]-0.07*(EXT[3]-EXT[2])],s=16,c="#1764d8",alpha=0.6,zorder=20)
axs.text(EXT[0]+0.085*(EXT[1]-EXT[0]),EXT[3]-0.07*(EXT[3]-EXT[2]),"training cameras",fontsize=9,va="center",zorder=20)
axs.scatter([EXT[0]+0.06*(EXT[1]-EXT[0])],[EXT[3]-0.12*(EXT[3]-EXT[2])],s=16,c="#ee7b16",alpha=0.92,edgecolors="#a14a00",linewidths=0.4,zorder=20)
axs.text(EXT[0]+0.085*(EXT[1]-EXT[0]),EXT[3]-0.12*(EXT[3]-EXT[2]),"held-out gap cameras",fontsize=9,va="center",zorder=20)

# ---- body: 9x5 floater grid ----
bg=gs[1].subgridspec(len(ORDER),len(ANGLES),wspace=0.04,hspace=0.10)
for i,m in enumerate(ORDER):
    for j,a in enumerate(ANGLES):
        ax=fig.add_subplot(bg[i,j]); g=grids[(m,a)]
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(np.clip(refS,0,1),origin="lower",extent=EXT,cmap="Greys",vmin=0,vmax=1.2,
                  alpha=0.30,aspect="equal",interpolation=INTERP)
        if g is not None:
            gs_=np.sqrt(np.clip(g.astype(float),0,None)/vmax)
            ax.imshow(np.clip(gs_,0,1),origin="lower",extent=EXT,cmap="inferno",vmin=0,vmax=1,
                      alpha=0.92,aspect="equal",interpolation=INTERP)
            fr=frac(m,a)
            if fr is not None:
                ax.text(0.96,0.04,f"{fr*100:.0f}%",color=("#39ff14" if m=="Ours" else "w"),
                        ha="right",va="bottom",transform=ax.transAxes,fontsize=10,fontweight="bold")
        else:
            ax.text(0.5,0.5,"n/a",color="gray",ha="center",transform=ax.transAxes)
        ax.set_xlim(EXT[0],EXT[1]); ax.set_ylim(EXT[2],EXT[3])
        # visible dashed orange gap outline on each panel, widening with the column's angle, so the
        # reader reads floaters relative to the marked gap -- the link to the header schematic.
        ga=geoms[a]; _gh=ga['gap_half']; _gd=ga['gap_dir']; _pr=half*1.25
        ax.add_patch(Wedge(tuple(ga['crop_center']),_pr,math.degrees(_gd-_gh),math.degrees(_gd+_gh),
                           facecolor="#ee7b16",alpha=0.08,edgecolor="none",zorder=5))
        ax.add_patch(Wedge(tuple(ga['crop_center']),_pr,math.degrees(_gd-_gh),math.degrees(_gd+_gh),
                           facecolor="none",edgecolor="#ee7b16",lw=1.5,linestyle="--",alpha=0.90,zorder=11))
        if i==0:
            # column header: angle + small wedge icon whose width = angle
            ax.set_title(f"{a}° gap",fontsize=11,fontweight="bold")
        if j==0:
            ax.set_ylabel(NICE[m],fontsize=10.5,fontweight=("bold" if m=="Ours" else "normal"),
                          color=("crimson" if m=="Ours" else "black"))
# colorbar
cax=fig.add_axes([0.918,0.10,0.012,0.46])
cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap="inferno"),cax=cax)
cb.set_ticks([0.03,0.97]); cb.set_ticklabels(["low\n(clean)","high\n(dense\nfloaters)"])
cb.ax.tick_params(labelsize=9)
cb.set_label("free-space floater density  ->  brighter = more floaters",fontsize=10)

fig.suptitle(f"Free-space floaters under biased viewpoints — {SCENE}  "
             f"(header: scene + camera gap; body: per-method floater density)",
             fontsize=14,y=0.997)
plt.subplots_adjust(left=0.055,right=0.905,top=0.955,bottom=0.045)
fig.savefig(OUT,dpi=150); plt.close(fig)
print("wrote",OUT)
