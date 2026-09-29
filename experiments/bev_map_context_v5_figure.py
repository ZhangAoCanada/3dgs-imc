#!/usr/bin/env python3
"""BEV map-context v5: readable source-camera layout plus robust PIL grid."""
import math, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Polygon, Wedge
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/data1/zhangao/repos/3dgs-imc-codex')
sys.path.insert(0,str(ROOT/'experiments'))
import bev_map_context_v4_figure as base

VM=base.VM; GA=base.GA; SEP=base.SEP
ANGLES=base.ANGLES; ORDER=base.ORDER; NICE=base.NICE
BLUE=base.BLUE; BLUE_FILL=base.BLUE_FILL; ORANGE=base.ORANGE; ORANGE_FILL=base.ORANGE_FILL; INK=base.INK
FONT=base.FONT
try:
    RESAMPLE=Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE=Image.LANCZOS

def font(size,bold=False):
    path=FONT if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try: return ImageFont.truetype(path,size)
    except OSError: return ImageFont.load_default()

def poly(center,f,r,pts):
    c=np.asarray(center,float)
    return np.asarray([c+x*f+y*r for x,y in pts])

def draw_cam(ax,xy,theta,color,fill,size,alpha=1.0,lw=.55,z=10):
    f=np.array([math.cos(theta),math.sin(theta)]); r=np.array([-math.sin(theta),math.cos(theta)])
    c=np.asarray(xy,float)
    fr=poly(c+f*size*.20,f,r,[(0,-.11*size),(.78*size,-.40*size),(.78*size,.40*size),(0,.11*size)])
    body=poly(c-f*size*.07,f,r,[(-.22*size,-.22*size),(-.22*size,.22*size),(.18*size,.22*size),(.18*size,-.22*size)])
    ax.add_patch(Polygon(fr,closed=True,facecolor=fill,edgecolor=color,lw=lw,alpha=.25*alpha,zorder=z,joinstyle='round'))
    ax.add_patch(Polygon(body,closed=True,facecolor='white',edgecolor=color,lw=lw,alpha=.86*alpha,zorder=z+1,joinstyle='round'))
    ax.add_patch(Circle(c-f*size*.06,radius=size*.038,facecolor=color,edgecolor='none',alpha=.90*alpha,zorder=z+2))

def camera_context_extent(geom):
    xy=geom['xy']; cc=geom['crop_center'][None,:]
    pts=np.vstack([xy,cc])
    mn=pts.min(0); mx=pts.max(0); span=mx-mn
    pad=0.18*max(float(span[0]),float(span[1]))
    mn-=pad; mx+=pad
    c=(mn+mx)/2; half=max(float(mx[0]-mn[0]),float(mx[1]-mn[1]))/2
    return [float(c[0]-half),float(c[0]+half),float(c[1]-half),float(c[1]+half)]

def adjusted_geoms(geoms):
    ctx=camera_context_extent(geoms[20])
    out={}
    for a,g in geoms.items():
        gg=dict(g); gg['ctx_ext']=ctx; out[a]=gg
    return out

def draw_context(ax,bg,crop_ext,g,angle,legend=False):
    ctx=g['ctx_ext']; cc=g['crop_center']; span=ctx[1]-ctx[0]
    ax.set_facecolor('#fbfaf7')
    corners=np.array([[ctx[0],ctx[2]],[ctx[0],ctx[3]],[ctx[1],ctx[2]],[ctx[1],ctx[3]]],float)
    radius=float(np.linalg.norm(corners-cc[None,:],axis=1).max()*1.08)
    t1=math.degrees(g['gap_dir']-g['gap_half']); t2=math.degrees(g['gap_dir']+g['gap_half'])
    ax.add_patch(Wedge(tuple(cc),radius,t1,t2,facecolor=ORANGE,edgecolor=ORANGE,lw=1.0,alpha=.15,zorder=1))

    # The RGB BEV map is an orientation inset. It is deliberately scaled visually, not by the huge BEV crop extent.
    inset_half=.205*span
    inset=[cc[0]-inset_half,cc[0]+inset_half,cc[1]-inset_half,cc[1]+inset_half]
    ax.imshow(np.asarray(bg)/255.0,origin='upper',extent=inset,aspect='equal',interpolation='bicubic',zorder=2)
    ax.plot([inset[0],inset[1],inset[1],inset[0],inset[0]],[inset[2],inset[2],inset[3],inset[3],inset[2]],color=INK,lw=.95,zorder=5)

    xy=g['xy']; th=g['theta']; mask=g['is_gap']
    train_size=.0105*span; gap_size=.0130*span
    for p,t in zip(xy[~mask],th[~mask]): draw_cam(ax,p,t,BLUE,BLUE_FILL,train_size,alpha=.52,lw=.38,z=9)
    for p,t in zip(xy[mask],th[mask]): draw_cam(ax,p,t,ORANGE,ORANGE_FILL,gap_size,alpha=.98,lw=.58,z=16)
    ax.add_patch(Circle(tuple(cc),radius=.0042*span,facecolor=INK,edgecolor='white',lw=.5,zorder=25))

    ax.text(.045,.945,f'{angle} deg held-out wedge',transform=ax.transAxes,ha='left',va='top',fontsize=9.4,fontweight='bold',color=INK,bbox=dict(facecolor='white',edgecolor='none',alpha=.84,pad=2.0),zorder=30)
    ax.text(.045,.055,f"all {g['n_all']} source cameras | gap {g['n_gap']}",transform=ax.transAxes,ha='left',va='bottom',fontsize=8.0,color=INK,bbox=dict(facecolor='white',edgecolor='none',alpha=.78,pad=1.8),zorder=30)
    if legend:
        x0=ctx[0]+.060*span; y0=ctx[3]-.180*span
        draw_cam(ax,[x0,y0],0,BLUE,BLUE_FILL,.034*span,1,.62,35)
        draw_cam(ax,[x0,y0-.075*span],0,ORANGE,ORANGE_FILL,.034*span,1,.68,35)
        ax.text(x0+.055*span,y0,'train',ha='left',va='center',fontsize=8.0,color=INK,zorder=36)
        ax.text(x0+.055*span,y0-.075*span,'gap',ha='left',va='center',fontsize=8.0,color=INK,zorder=36)
    ax.set_xlim(ctx[0],ctx[1]); ax.set_ylim(ctx[2],ctx[3]); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(.75); s.set_edgecolor('#3a3a3a')

def save_contexts(scene,crop_ext,geoms,bgs,bg_sources,ref_path):
    outdir=SEP/f'{scene}_mapcontext_v5'; outdir.mkdir(parents=True,exist_ok=True)
    lines=[f'Map-context v5 exports for BEV_overlay_{scene}_mapcontext_v5.png','',
           'Fix from v4: camera context extent is based on true source camera positions, not the much larger cropped BEV map extent.',
           'All cameras are still plotted from DATA/mipnerf360/bonsai/sparse/0/images.bin.',
           'Held-out membership comes from dataset split files. Coverage uses one continuous wedge, not arc rings.',
           'The Ours BEV RGB map is a visual orientation inset centered at the scene; it is not drawn at metric crop scale because that scale made the camera layout unreadable.','',
           f'BEV crop reference: {ref_path}',f'BEV metric crop extent retained for body panels: {[round(float(x),6) for x in crop_ext]}','']
    for a in ANGLES:
        fig,ax=plt.subplots(figsize=(4.2,4.2),dpi=220)
        draw_context(ax,bgs[a],crop_ext,geoms[a],a,legend=(a==ANGLES[0]))
        fig.subplots_adjust(0,0,1,1)
        fig.savefig(outdir/f'camera_context_{a}.png',dpi=220)
        plt.close(fig)
        bgs[a].save(outdir/f'ours_bev_inset_source_{a}.png')
        g=geoms[a]
        lines += [f'{a} deg',f"  all cameras plotted: {g['n_all']}",f"  train/gap cameras: {g['n_train']} / {g['n_gap']}",f"  split source: {g['split']}",f"  Ours BEV inset source: {bg_sources[a]}",f"  readable camera context extent: {[round(float(x),6) for x in g['ctx_ext']]}",f"  gap_dir_rad_from_true_gap_cameras: {g['gap_dir']:.8f}",f"  gap_half_rad_declared: {g['gap_half']:.8f}",'']
    (outdir/'README.txt').write_text('\n'.join(lines)+'\n')
    return outdir

def text_center(draw,box,text,ft,fill):
    x0,y0,x1,y1=box
    b=draw.textbbox((0,0),text,font=ft)
    draw.text((x0+(x1-x0-(b[2]-b[0]))/2,y0+(y1-y0-(b[3]-b[1]))/2),text,font=ft,fill=fill)

def annotate_pct(img,scene,method,angle):
    fr=base.frac(scene,method,angle)
    if fr is None: return img
    out=img.copy(); d=ImageDraw.Draw(out)
    txt=f'{fr*100:.0f}%'; ft=font(22,True)
    fill=(57,255,20) if method=='Ours' else (255,255,255)
    b=d.textbbox((0,0),txt,font=ft); x=out.width-(b[2]-b[0])-10; y=out.height-(b[3]-b[1])-10
    d.text((x+1,y+1),txt,font=ft,fill=(0,0,0)); d.text((x,y),txt,font=ft,fill=fill)
    return out

def make_colorbar(h,w=22):
    cmap=plt.get_cmap('inferno')
    vals=np.linspace(1,0,h)[:,None]
    rgb=(cmap(vals)[...,:3]*255).astype(np.uint8).reshape(h,1,3)
    return Image.fromarray(np.repeat(rgb,w,axis=1),'RGB')

def compose(scene,context_dir):
    cell=360; gap=8; left=158; right=122; top=78; footer=58
    rows=1+len(ORDER); cols=len(ANGLES)
    width=left+cols*cell+(cols-1)*gap+right
    height=top+rows*cell+(rows-1)*gap+footer
    canvas=Image.new('RGB',(width,height),(255,255,255)); d=ImageDraw.Draw(canvas)
    d.text((left,18),'Free-space floater density with source-defined camera gaps - bonsai',font=font(22,True),fill=INK)
    for c,a in enumerate(ANGLES):
        x=left+c*(cell+gap); text_center(d,(x,42,x+cell,72),f'{a} deg',font(18,True),INK)
    y0=top
    d.text((18,y0+cell//2-28),'Camera layout\n(source split)',font=font(16,True),fill=INK,spacing=4)
    for c,a in enumerate(ANGLES):
        img=Image.open(context_dir/f'camera_context_{a}.png').convert('RGB').resize((cell,cell),RESAMPLE)
        canvas.paste(img,(left+c*(cell+gap),y0))
    for r,m in enumerate(ORDER,start=1):
        y=top+r*(cell+gap)
        label=NICE[m]; fill=(180,30,45) if m=='Ours' else INK
        b=d.multiline_textbbox((0,0),label,font=font(17, m=='Ours'))
        d.text((left-18-(b[2]-b[0]),y+cell//2-(b[3]-b[1])//2),label,font=font(17,m=='Ours'),fill=fill)
        for c,a in enumerate(ANGLES):
            img,_=base.load_panel(scene,m,a)
            img=img.resize((cell,cell),RESAMPLE)
            img=annotate_pct(img,scene,m,a)
            canvas.paste(img,(left+c*(cell+gap),y))
    body_y=top+cell+gap; body_h=len(ORDER)*cell+(len(ORDER)-1)*gap
    cb=make_colorbar(body_h,24); cbx=left+cols*cell+(cols-1)*gap+32
    canvas.paste(cb,(cbx,body_y))
    d.text((cbx+34,body_y),'high\n(dense\nfloaters)',font=font(13),fill=INK,spacing=2)
    d.text((cbx+34,body_y+body_h-48),'low\n(clean)',font=font(13),fill=INK,spacing=2)
    d.text((left,height-38),'Top row: all source cameras with continuous held-out wedge and Ours BEV RGB inset. Body: brighter heat = denser free-space floaters.',font=font(14),fill=INK)
    out=GA/f'BEV_overlay_{scene}_mapcontext_v5.png'; canvas.save(out)
    return out

def main():
    scene='bonsai'
    print('v5 CPU-only generator: readable source-camera context + PIL grid',flush=True)
    center,axes,crop_ext,ref_path=base.crop_extent(scene)
    geoms=adjusted_geoms(base.build_geoms(scene,center,axes,crop_ext))
    print('camera extent', [round(float(x),3) for x in geoms[20]['ctx_ext']], 'all cameras', geoms[20]['n_all'], flush=True)
    bgs={}; bg_sources={}
    for a in ANGLES: bgs[a],bg_sources[a]=base.load_bg(scene,a)
    outdir=save_contexts(scene,crop_ext,geoms,bgs,bg_sources,ref_path)
    out=compose(scene,outdir)
    print('wrote',out,flush=True); print('separate',outdir,flush=True)

if __name__=='__main__': main()
