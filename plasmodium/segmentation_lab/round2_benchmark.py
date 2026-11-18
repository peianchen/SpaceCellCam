"""Repeatable original-coordinate round1/round2 comparison on raw PNG manifests."""
from __future__ import annotations
import argparse, csv, hashlib, html, json, shutil, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from .contracts import RoiSpec
from .segmenter import segment as round1
from .segmenter_round2 import segment as round2, DEFAULTS, prepare_baseline
from .visualize import write_image,mask_overlay,save_method_images,_font,_roi_crop,_square_rgb


def pixel_hash(image):
    return hashlib.sha256(f"BGR|{image.shape}|{image.dtype}".encode()+np.ascontiguousarray(image).tobytes()).hexdigest()


def difference_overlay(image, old, new):
    result=mask_overlay(image,old,alpha=.06)
    added=new & ~old
    result[added]=np.rint(.35*result[added]+.65*np.array([0,165,255])).astype(np.uint8)
    return result


def compare_figure(path, image, before, after, roi, title, crop=None):
    pictures=[image,mask_overlay(image,before.mask,excluded_food=before.excluded_food),
              mask_overlay(image,after.mask,excluded_food=after.excluded_food),difference_overlay(image,before.mask,after.mask)]
    width=540; canvas=Image.new('RGB',(width*4,width+154),'#f5f7fa');draw=ImageDraw.Draw(canvas)
    draw.text((15,10),title,font=_font(26,True),fill='#243746')
    draw.text((15,50),'綠色：辨識邊界｜紫色：食物排除邊界｜橘色：本輪新增像素（仍需人工確認）',font=_font(22),fill='#435563')
    labels=['原始影像','第一輪保留結果','第二輪：局部細網','差異：新增範圍']
    for i,picture in enumerate(pictures):
        draw.text((i*width+12,93),labels[i],font=_font(23,True),fill='#243746')
        if crop:
            x0,y0,x1,y1=crop; part=picture[y0:y1,x0:x1]
        else:part=_roi_crop(picture,roi)
        canvas.paste(_square_rgb(part,width),(i*width,144))
    canvas.save(path,quality=94)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifests',nargs='+',type=Path,required=True)
    p.add_argument('--geometry',type=Path,default=Path('segmentation_lab/config_round1_v2.json'))
    p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review-crops',type=Path);p.add_argument('--ablation',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text(encoding='utf-8')) if a.config else DEFAULTS
    geom=json.loads(a.geometry.read_text(encoding='utf-8'));rows=[];start=time.time()
    crops=json.loads(a.review_crops.read_text(encoding='utf-8')) if a.review_crops else {}
    if 'samples' in crops:crops={x['sample_id']:x['review_crop_xyxy'] for x in crops['samples']}
    crops.update({'rotating_02_b3ec4905_late_f006390':[650,1170,1100,1530],
                  'rotating_03_3636b20c_late_f006231':[850,1150,1250,1550],
                  'still_00_31bd0a24_late_f006627':[1000,600,1450,1050]})
    frozen={'parameters':cfg,'base_config':geom['methods']['adaptive_chroma'],'source_hashes':{},'input_manifests':[],'ground_truth':None}
    for name in ['segmenter_round2.py','segmenter.py','round2_benchmark.py','contracts.py']:
        path=Path(__file__).parent/name;frozen['source_hashes'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output/'source').mkdir()
    for name in frozen['source_hashes']:shutil.copy2(Path(__file__).parent/name,a.output/'source'/name)
    (a.output/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (a.output/'geometry.json').write_text(json.dumps(geom,indent=2),encoding='utf-8')
    for manifest in a.manifests:
        data=json.loads(manifest.read_text(encoding='utf-8'))
        frozen['input_manifests'].append({'path':str(manifest.resolve()),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
        for source in data['samples']:
            sid=source['sample_id'];out=a.output/'samples'/sid;out.mkdir(parents=True)
            image=cv2.imdecode(np.fromfile(manifest.parent/source['image_path'],np.uint8),cv2.IMREAD_COLOR)
            if image is None or pixel_hash(image)!=source['source_pixel_sha256']:raise ValueError(f'Raw pixel hash mismatch: {sid}')
            stat=Path(source['source_video']).stat()
            # Metadata names differ in earlier manifests; verify those present.
            for key,value in [('source_file_bytes',stat.st_size),('source_file_mtime_ns',stat.st_mtime_ns)]:
                if key in source and source[key]!=value:raise ValueError(f'Raw video changed: {sid} {key}')
            write_image(out/'original.png',image)
            r=dict(geom['roi']);r.update(geom['new_method_roi_by_video'][source['video_id']]);roi=RoiSpec(**r)
            before=prepare_baseline(image,roi,base_config=geom['methods']['adaptive_chroma']);after=round2(image,roi,config=cfg,base_config=geom['methods']['adaptive_chroma'],baseline=before)
            record={k:source.get(k) for k in ['sample_id','source_video','video_id','frame_index','section','group','evaluation_role','source_pixel_sha256']}
            record.update({'original_path':f'samples/{sid}/original.png','image_path':f'samples/{sid}/original.png','roi':r,
                 'round1_area_px':int(before.mask.sum()),'round2_area_px':int(after.mask.sum()),
                 'added_px':int((after.mask&~before.mask).sum()),'removed_px':int((before.mask&~after.mask).sum()),
                 'qc_flags':after.qc_flags,'diagnostics':after.diagnostics,'accuracy_metrics':None,
                 'comparison_path':f'samples/{sid}/comparison.jpg'})
            if 'methods' in source and 'adaptive_chroma' in source['methods']:
                oldpath=manifest.parent/source['methods']['adaptive_chroma']['mask_path']
                old=cv2.imdecode(np.fromfile(oldpath,np.uint8),0)>0
                record['round1_replay_bit_equal']=bool(np.array_equal(before.mask,old))
                if not record['round1_replay_bit_equal']:raise ValueError(f'Baseline replay changed: {sid}')
            for name,result in [('round1',before),('round2',after)]:
                paths=save_method_images(out/name,image,result)
                (out/name/'qc.json').write_text(json.dumps({'qc_flags':result.qc_flags,'diagnostics':result.diagnostics},indent=2),encoding='utf-8')
                record[name+'_mask_path']=f'samples/{sid}/{name}/{paths["mask_path"]}'
            write_image(out/'additions.png',(after.mask&~before.mask).astype(np.uint8)*255)
            title=f'{Path(source["source_video"]).name} | 影格 {source["frame_index"]}'
            compare_figure(out/'comparison.jpg',image,before,after,roi,title)
            crop=crops.get(sid)
            if isinstance(crop,dict):crop=crop.get('review_crop_xyxy')
            if crop:
                record['review_crop_xyxy']=crop;compare_figure(out/'detail.jpg',image,before,after,roi,title+' | 同位置放大',crop)
            if a.ablation:
                naive=round1(image,roi,config={**geom['methods']['adaptive_chroma'],'low_evidence':.018,'high_evidence':.045,'noise_multiplier':.3})
                save_method_images(out/'lower_threshold_only',image,naive);record['ablation_area_px']=int(naive.mask.sum())
            (out/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');rows.append(record)
            print(sid,'added',record['added_px'],flush=True)
    frozen.update({'samples':rows,'sample_count':len(rows),'elapsed_seconds':time.time()-start,'accuracy_metrics':None,'no_temporal_mask_editing':True})
    (a.output/'results.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    with (a.output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['sample_id','evaluation_role','round1_area_px','round2_area_px','added_px','removed_px','ablation_area_px']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    cards=[]
    for r in rows:
        folder='samples/'+r['sample_id'];links=' '.join(f'<a href="{folder}/{p}">{label}</a>' for p,label in [('original.png','原圖'),('round1/overlay.png','第一輪原尺寸'),('round2/overlay.png','第二輪原尺寸'),('round2/mask.png','二值遮罩'),('round2/preprocessed.png','明暗校正顯示'),('round2/score.png','特徵強度'),('round2/qc.json','QC')])
        detail=f'<img loading="lazy" src="{folder}/detail.jpg">' if r.get('review_crop_xyxy') else ''
        cards.append(f'<section><h2>{html.escape(r["sample_id"])}</h2><p>{links}</p><p>新增 {r["added_px"]:,} 像素；新增並不代表全數正確。QC：{html.escape(", ".join(r["qc_flags"]))}</p><img loading="lazy" src="{folder}/comparison.jpg">{detail}</section>')
    page='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>黏菌第二輪比較</title><style>body{font:18px sans-serif;margin:24px;background:#f5f7fa;color:#243746}img{width:100%;height:auto}section{background:white;padding:16px;margin:24px 0}a{margin-right:16px}h1{font-size:32px}</style><h1>同一原始影像：第一輪與第二輪</h1><p>綠色＝辨識邊界；紫色＝食物排除；橘色＝本輪新增。沒有人工真值，這是視覺比較，不是準確率驗證。空白原圖、失敗案例與QC均保留。</p>'+''.join(cards)+'</html>'
    (a.output/'gallery.html').write_text(page,encoding='utf-8')
    print(json.dumps({'samples':len(rows),'seconds':round(time.time()-start,2)}),flush=True)


if __name__=='__main__':main()
