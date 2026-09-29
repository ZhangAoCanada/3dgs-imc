#!/usr/bin/env python3
import json, math, struct, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
from PIL import Image, ImageFont
from scipy.ndimage import uniform_filter

VM=Path('/data1/zhangao/outputs/viz_multiangle')
GA=VM/'global_all'
SEP=VM/'SEPARATE'/'BEV'
BASE=Path('/data1/zhangao/outputs/baselines_multi_angle')
DATA=Path('/data1/zhangao/DATA/mipnerf360')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
ANGLES=[20,40,60,80,100]
ORDER=['3DGS','Hier','ScaffoldGS','MCMC','CDGS','PGSR','MH-3DGS','MVGS','Ours']
NICE={'3DGS':'3DGS','Hier':'Hier-GS','ScaffoldGS':'Scaffold-GS','MCMC':'3DGS-MCMC','CDGS':'CDGS','PGSR':'PGSR','MH-3DGS':'MH-3DGS','MVGS':'MVGS','Ours':'Ours (GS-LSS)'}
BLUE='#1764d8'; BLUE_FILL='#dceaff'; ORANGE='#ee7b16'; ORANGE_FILL='#ffe0b8'; INK='#202124'
try:
    RESAMPLE=Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE=Image.BICUBIC

def safe(name): return str(name).replace('/','-')

def qvec2rotmat(q):
    q=np.asarray(q,dtype=np.float64); q=q/max(np.linalg.norm(q),1e-12)
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*z*x+2*w*y],
                     [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
                     [2*z*x-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]],dtype=np.float64)

def read_images_bin_pose_only(path):
    rec=[]
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            data=struct.unpack('<idddddddi',f.read(64))
            q=np.array(data[1:5]); t=np.array(data[5:8])
            bs=bytearray()
            while True:
                c=f.read(1)
                if not c: raise EOFError('unexpected EOF in image name')
                if c==b'\x00': break
                bs.extend(c)
            name=bs.decode('utf-8',errors='replace')
            n2=struct.unpack('<Q',f.read(8))[0]
            f.seek(24*n2,1)
            R=qvec2rotmat(q)
            center=-(R.T@t)
            forward=R.T@np.array([0.0,0.0,1.0])
            rec.append({'name':name,'stem':Path(name).stem,'center':center.astype(np.float32),'forward':forward.astype(np.float32)})
    return rec

def read_ply_xyz(path):
    with open(path,'rb') as f:
        header=[]
        while True:
            line=f.readline()
            if not line: raise EOFError('no end_header')
            s=line.decode('ascii',errors='replace').strip(); header.append(s)
            if s=='end_header': break
        if 'format binary_little_endian 1.0' not in header: raise ValueError('expected binary_little_endian ply')
        n=None; props=[]; in_v=False
        for s in header:
            parts=s.split()
            if len(parts)>=3 and parts[:2]==['element','vertex']:
                n=int(parts[2]); in_v=True; continue
            if len(parts)>=2 and parts[0]=='element' and parts[1]!='vertex': in_v=False
            if in_v and len(parts)==3 and parts[0]=='property': props.append((parts[1],parts[2]))
        mp={'float':'<f4','float32':'<f4','double':'<f8','uchar':'u1','uint8':'u1','char':'i1','int8':'i1','ushort':'<u2','uint16':'<u2','short':'<i2','int16':'<i2','uint':'<u4','uint32':'<u4','int':'<i4','int32':'<i4'}
        dt=np.dtype([(name,mp[typ]) for typ,name in props])
        arr=np.fromfile(f,dtype=dt,count=n)
    return np.stack([arr['x'],arr['y'],arr['z']],1).astype(np.float32)

def proj_from_ref(ref):
    lo=np.percentile(ref,1,0); hi=np.percentile(ref,99,0)
    center=ref.mean(0); order=np.argsort(-(hi-lo))[:2]
    axes=np.zeros((2,3),np.float32)
    for i,a in enumerate(order): axes[i,a]=1.0
    return center.astype(np.float32),axes

def main_box_proj(ref,center,axes,mass=0.88,margin=2.8):
    xy=(ref-center)@axes.T
    H,xe,ye=np.histogram2d(xy[:,0],xy[:,1],bins=200)
    Hs=uniform_filter(H,size=9)
    pi,pj=np.unravel_index(np.argmax(Hs),Hs.shape)
    cx=(xe[pi]+xe[pi+1])/2; cy=(ye[pj]+ye[pj+1])/2
    d=np.maximum(np.abs(xy[:,0]-cx),np.abs(xy[:,1]-cy))
    half=float(np.quantile(d,mass))*margin
    return float(cx),float(cy),half

def crop_extent(scene):
    paths=sorted((BASE/scene).glob('*/input.ply'))
    if not paths: raise FileNotFoundError(BASE/scene)
    ref=read_ply_xyz(paths[0]); center,axes=proj_from_ref(ref)
    cx,cy,half=main_box_proj(ref,center,axes)
    return center,axes,[cx-half,cx+half,cy-half,cy+half],paths[0]

def split_names(scene,angle):
    path=DATA/scene/f'test_selection_angle_{angle}.0.txt'
    names={Path(x.strip()).stem for x in path.read_text().splitlines() if x.strip()}
    return names,path

def circular_mean(a):
    a=np.asarray(a,dtype=float)
    return float(math.atan2(np.sin(a).mean(),np.cos(a).mean()))

def context_extent(xy,crop_ext):
    crop=np.array([[crop_ext[0],crop_ext[2]],[crop_ext[0],crop_ext[3]],[crop_ext[1],crop_ext[2]],[crop_ext[1],crop_ext[3]]],float)
    pts=np.vstack([xy,crop]); mn=pts.min(0); mx=pts.max(0); span=mx-mn
    pad=0.10*max(float(span[0]),float(span[1])); mn-=pad; mx+=pad
    c=(mn+mx)/2; half=max(float(mx[0]-mn[0]),float(mx[1]-mn[1]))/2
    return [c[0]-half,c[0]+half,c[1]-half,c[1]+half]

def build_geoms(scene,center,axes,crop_ext):
    images_path=DATA/scene/'sparse/0/images.bin'
    rec=read_images_bin_pose_only(images_path)
    names=[r['stem'] for r in rec]
    xyz=np.stack([r['center'] for r in rec]); fwd=np.stack([r['forward'] for r in rec])
    xy=(xyz-center)@axes.T; view_xy=fwd@axes.T
    cc=np.array([(crop_ext[0]+crop_ext[1])/2,(crop_ext[2]+crop_ext[3])/2],float)
    to_center=cc[None,:]-xy
    valid=np.linalg.norm(view_xy,axis=1)>1e-7
    if valid.any() and np.median(np.sum(view_xy[valid]*to_center[valid],axis=1))<0: view_xy=-view_xy
    bad=np.linalg.norm(view_xy,axis=1)<=1e-7
    view_xy[bad]=to_center[bad]
    theta=np.arctan2(view_xy[:,1],view_xy[:,0])
    cam_ang=np.arctan2(xy[:,1]-cc[1],xy[:,0]-cc[0])
    ctx=context_extent(xy,crop_ext)
    out={}
    for angle in ANGLES:
        held,path=split_names(scene,angle)
        missing=sorted(held.difference(names))
        if missing: raise RuntimeError(f'{len(missing)} split names did not match images.bin: {missing[:5]}')
        is_gap=np.array([n in held for n in names],bool)
        out[angle]={'xy':xy,'theta':theta,'angles':cam_ang,'is_gap':is_gap,'gap_dir':circular_mean(cam_ang[is_gap]),'gap_half':math.radians(angle/2),'crop_center':cc,'ctx_ext':ctx,'images_bin':images_path,'split':path,'n_all':len(names),'n_train':int((~is_gap).sum()),'n_gap':int(is_gap.sum())}
    return out

def load_bg(scene,angle):
    cands=[SEP/f'{scene}_mapcontext_v3'/f'ours_bev_background_{angle}.png',SEP/f'{scene}_mapcontext_v2'/f'ours_bev_background_{angle}.png',SEP/f'{scene}_mapcontext'/f'ours_bev_background_{angle}.png']
    for p in cands:
        if p.exists(): return Image.open(p).convert('RGB'),p
    raise FileNotFoundError(cands)

def frac(scene,method,angle):
    p=GA/f'{scene}__{method}.frac.json'
    if not p.exists(): return None
    d=json.loads(p.read_text())
    return d.get(str(angle)) or d.get(str(float(angle)))

def load_panel(scene,method,angle):
    p=SEP/scene/f'{safe(NICE[method])}_{angle}.png'
    img=Image.open(p).convert('RGB')
    w,h=img.size
    if h>w: img=img.crop((0,h-w,w,h))
    return img,p

def poly(center,f,r,pts):
    c=np.asarray(center,float)
    return np.asarray([c+x*f+y*r for x,y in pts])

def draw_cam(ax,xy,theta,color,fill,size,alpha=1,lw=.6,z=8):
    f=np.array([math.cos(theta),math.sin(theta)]); r=np.array([-math.sin(theta),math.cos(theta)])
    c=np.asarray(xy,float)
    fr=poly(c+f*size*.20,f,r,[(0,-.12*size),(.76*size,-.42*size),(.76*size,.42*size),(0,.12*size)])
    body=poly(c-f*size*.07,f,r,[(-.22*size,-.23*size),(-.22*size,.23*size),(.18*size,.23*size),(.18*size,-.23*size)])
    ax.add_patch(Polygon(fr,closed=True,facecolor=fill,edgecolor=color,lw=lw,alpha=.28*alpha,zorder=z,joinstyle='round'))
    ax.add_patch(Polygon(body,closed=True,facecolor='white',edgecolor=color,lw=lw,alpha=.90*alpha,zorder=z+1,joinstyle='round'))
    ax.add_patch(Circle(c-f*size*.06,radius=size*.04,facecolor=color,edgecolor='none',alpha=.9*alpha,zorder=z+2))

def draw_context(ax,bg,crop_ext,g,angle,legend=False):
    ctx=g['ctx_ext']; cc=g['crop_center']; ax.set_facecolor('#fbfaf7')
    corners=np.array([[ctx[0],ctx[2]],[ctx[0],ctx[3]],[ctx[1],ctx[2]],[ctx[1],ctx[3]]])
    radius=float(np.linalg.norm(corners-cc[None,:],axis=1).max()*1.08)
    t1=math.degrees(g['gap_dir']-g['gap_half']); t2=math.degrees(g['gap_dir']+g['gap_half'])
    ax.add_patch(Wedge(tuple(cc),radius,t1,t2,facecolor=ORANGE,edgecolor=ORANGE,lw=1.0,alpha=.16,zorder=1))
    ax.imshow(np.asarray(bg)/255.0,origin='upper',extent=crop_ext,aspect='equal',interpolation='bicubic',zorder=2)
    ax.add_patch(Rectangle((crop_ext[0],crop_ext[2]),crop_ext[1]-crop_ext[0],crop_ext[3]-crop_ext[2],facecolor='none',edgecolor=INK,lw=1.15,zorder=5))
    span=ctx[1]-ctx[0]; train_size=.0105*span; gap_size=.0120*span
    xy=g['xy']; th=g['theta']; mask=g['is_gap']
    for p,t in zip(xy[~mask],th[~mask]): draw_cam(ax,p,t,BLUE,BLUE_FILL,train_size,alpha=.55,lw=.42,z=9)
    for p,t in zip(xy[mask],th[mask]): draw_cam(ax,p,t,ORANGE,ORANGE_FILL,gap_size,alpha=.98,lw=.62,z=14)
    ax.add_patch(Circle(tuple(cc),radius=.0045*span,facecolor=INK,edgecolor='white',lw=.5,zorder=20))
    ax.text(.045,.945,f'{angle} deg held-out wedge',transform=ax.transAxes,ha='left',va='top',fontsize=9.2,fontweight='bold',color=INK,bbox=dict(facecolor='white',edgecolor='none',alpha=.84,pad=2),zorder=30)
    ax.text(.045,.055,f"all {g['n_all']} cameras | train {g['n_train']} | gap {g['n_gap']}",transform=ax.transAxes,ha='left',va='bottom',fontsize=7.7,color=INK,bbox=dict(facecolor='white',edgecolor='none',alpha=.78,pad=1.8),zorder=30)
    if legend:
        x0=ctx[0]+.058*span; y0=ctx[3]-.17*span
        draw_cam(ax,[x0,y0],0,BLUE,BLUE_FILL,.030*span,1,.75,32); draw_cam(ax,[x0,y0-.070*span],0,ORANGE,ORANGE_FILL,.030*span,1,.82,32)
        ax.text(x0+.050*span,y0,'train',ha='left',va='center',fontsize=7.8,color=INK,zorder=33)
        ax.text(x0+.050*span,y0-.070*span,'gap',ha='left',va='center',fontsize=7.8,color=INK,zorder=33)
    ax.set_xlim(ctx[0],ctx[1]); ax.set_ylim(ctx[2],ctx[3]); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(.75); s.set_edgecolor('#3a3a3a')

def save_context_assets(scene,crop_ext,geoms,bgs,bg_sources,ref_path):
    outdir=SEP/f'{scene}_mapcontext_v4'; outdir.mkdir(parents=True,exist_ok=True)
    lines=[f'Map-context v4 exports for BEV_overlay_{scene}_mapcontext_v4.png','', 'This version plots every dataset camera from COLMAP images.bin and uses split text files directly.', 'The held-out coverage is the older continuous sector/wedge; no separated arc-ring schematic is used.','',f'camera pose source: {DATA/scene/"sparse/0/images.bin"}',f'BEV crop reference: {ref_path}',f'BEV crop extent [xmin,xmax,ymin,ymax]: {[round(float(x),6) for x in crop_ext]}','']
    for angle in ANGLES:
        fig,ax=plt.subplots(figsize=(4.2,4.2)); draw_context(ax,bgs[angle],crop_ext,geoms[angle],angle,legend=(angle==ANGLES[0])); fig.tight_layout(pad=.02)
        fig.savefig(outdir/f'camera_context_{angle}.png',dpi=220); plt.close(fig)
        bgs[angle].save(outdir/f'ours_bev_background_{angle}.png')
        g=geoms[angle]
        lines += [f'{angle} deg',f"  all cameras plotted: {g['n_all']}",f"  train/gap cameras: {g['n_train']} / {g['n_gap']}",f"  split source: {g['split']}",f"  Ours BEV background source: {bg_sources[angle]}",f"  context extent [xmin,xmax,ymin,ymax]: {[round(float(x),6) for x in g['ctx_ext']]}",f"  gap_dir_rad_from_true_gap_cameras: {g['gap_dir']:.8f}",f"  gap_half_rad_declared: {g['gap_half']:.8f}",'']
    (outdir/'README.txt').write_text('\n'.join(lines)+'\n')
    return outdir

def draw_composite(scene,crop_ext,geoms,bgs):
    fig,axes=plt.subplots(len(ORDER)+1,len(ANGLES),figsize=(13.6,24.8),gridspec_kw={'height_ratios':[1.32]+[1]*len(ORDER),'hspace':.035,'wspace':.035})
    panel_sources={}
    for col,angle in enumerate(ANGLES):
        ax=axes[0,col]; draw_context(ax,bgs[angle],crop_ext,geoms[angle],angle,legend=(col==0)); ax.set_title(f'{angle} deg',fontsize=12.4,fontweight='bold',pad=4)
        if col==0: ax.set_ylabel('Camera layout\n(full source split)',fontsize=10.6,fontweight='bold',color=INK)
    for row,method in enumerate(ORDER,start=1):
        for col,angle in enumerate(ANGLES):
            ax=axes[row,col]; img,p=load_panel(scene,method,angle); panel_sources[(method,angle)]=p; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            fr=frac(scene,method,angle)
            if fr is not None: ax.text(.955,.045,f'{fr*100:.0f}%',transform=ax.transAxes,ha='right',va='bottom',fontsize=10.4,fontweight='bold',color=('#39ff14' if method=='Ours' else 'white'))
            for s in ax.spines.values(): s.set_linewidth(.45); s.set_edgecolor('#2f2f2f')
            if col==0: ax.set_ylabel(NICE[method],fontsize=10.8,fontweight=('bold' if method=='Ours' else 'normal'),color=('crimson' if method=='Ours' else INK))
    fig.suptitle('Free-space floater density with source-defined camera gaps - bonsai',fontsize=13.4,y=.996)
    fig.text(.5,.014,'Top row: all COLMAP dataset cameras and the continuous held-out wedge; black square marks the BEV crop used below. Body: brighter heat = denser free-space floaters.',ha='center',va='bottom',fontsize=9.8,color=INK)
    plt.tight_layout(rect=[.035,.030,.905,.986])
    cax=fig.add_axes([.922,.148,.014,.60]); cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='inferno'),cax=cax)
    cb.set_ticks([.03,.97]); cb.set_ticklabels(['low\n(clean)','high\n(dense\nfloaters)']); cb.ax.tick_params(labelsize=9); cb.set_label('free-space floater density -> brighter = more floaters',fontsize=10)
    out=GA/f'BEV_overlay_{scene}_mapcontext_v4.png'; fig.savefig(out,dpi=170); plt.close(fig); return out,panel_sources

def main():
    scene='bonsai'
    print('CPU-only v4 generator: no torch/project imports',flush=True)
    center,axes,crop_ext,ref_path=crop_extent(scene); print('crop', [round(float(x),4) for x in crop_ext], flush=True)
    geoms=build_geoms(scene,center,axes,crop_ext); print('cameras', geoms[20]['n_all'], flush=True)
    bgs={}; bg_sources={}
    for angle in ANGLES: bgs[angle],bg_sources[angle]=load_bg(scene,angle)
    outdir=save_context_assets(scene,crop_ext,geoms,bgs,bg_sources,ref_path)
    out,panel_sources=draw_composite(scene,crop_ext,geoms,bgs)
    with (outdir/'README.txt').open('a') as f:
        f.write('BEV panel sources used in composite:\n')
        for method in ORDER:
            for angle in ANGLES: f.write(f'  {NICE[method]} {angle} deg: {panel_sources[(method,angle)]}\n')
    print('wrote',out,flush=True); print('separate',outdir,flush=True)

if __name__=='__main__': main()
