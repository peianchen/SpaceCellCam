"""R2/R3 comparison on immutable raw-image manifests; no human accuracy claims."""
from __future__ import annotations
import argparse, csv, hashlib, html, json, shutil, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .contracts import RoiSpec
from .segmenter_round2 import prepare_baseline
from .segmenter_round3 import segment, DEFAULTS
from .round2_benchmark import pixel_hash
from .visualize import write_image, mask_overlay, save_method_images, _font, _roi_crop, _square_rgb


def comparison(path, image, before, after, roi, title, crop=None):
    diff=image.copy()
    added=after.mask & ~before.mask
    removed=before.mask & ~after.mask
    diff[added]=np.rint(.3*diff[added]+.7*np.array([0,165,255])).astype(np.uint8)
    diff[removed]=np.rint(.3*diff[removed]+.7*np.array([255,125,20])).astype(np.uint8)
    panels=[image, mask_overlay(image,before.mask,excluded_food=before.excluded_food),
            mask_overlay(image,after.mask,excluded_food=after.excluded_food),diff]
    width=540
    canvas=Image.new('RGB',(4*width,width+160),'#f5f7fa');draw=ImageDraw.Draw(canvas)
    draw.text((14,10),title,font=_font(25,True),fill='#243746')
    draw.text((14,51),'綠：辨識邊界｜紫：食物排除｜橘：新增｜藍：移除；差異不等於正確答案',font=_font(22),fill='#435563')
    for i,(panel,label) in enumerate(zip(panels,['原始影像','R2：前次保留結果','R3：方向性細網候選','變更位置：待人工確認'])):
        draw.text((i*width+12,96),label,font=_font(23,True),fill='#243746')
        part=panel[crop[1]:crop[3],crop[0]:crop[2]] if crop else _roi_crop(panel,roi)
        canvas.paste(_square_rgb(part,width),(i*width,150))
    canvas.save(path,quality=94)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifests',nargs='+',type=Path,required=True)
    p.add_argument('--geometry',type=Path,default=Path('segmentation_lab/config_round1_v2.json'))
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ids',nargs='*')
    a=p.parse_args();cv2.setNumThreads(2)
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text(encoding='utf-8')) if a.config else dict(DEFAULTS)
    geom=json.loads(a.geometry.read_text(encoding='utf-8'))
    source_dir=Path(__file__).parent; (a.output/'source').mkdir()
    frozen={'parameters':cfg,'base_config':geom['methods']['adaptive_chroma'],
            'ground_truth':None,'accuracy_metrics':None,'source_hashes':{},'input_manifests':[],
            'runtime':{'opencv':cv2.__version__,'numpy':np.__version__},'no_temporal_mask_editing':True}
    for name in ['segmenter_round3.py','segmenter_round2.py','segmenter.py','round3_benchmark.py','round2_benchmark.py','contracts.py','visualize.py']:
        frozen['source_hashes'][name]=hashlib.sha256((source_dir/name).read_bytes()).hexdigest()
        shutil.copy2(source_dir/name,a.output/'source'/name)
    for name,data in [('config.json',cfg),('geometry.json',geom)]:
        (a.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[];seen=set();start=time.time()
    for manifest in a.manifests:
        data=json.loads(manifest.read_text(encoding='utf-8'))
        frozen['input_manifests'].append({'path':str(manifest.resolve()),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
        for src in data['samples']:
            sid=src['sample_id']
            if a.ids and sid not in a.ids:continue
            if sid in seen:raise ValueError(f'Duplicate sample: {sid}')
            seen.add(sid);t=time.time()
            image=cv2.imdecode(np.fromfile(manifest.parent/src['image_path'],np.uint8),cv2.IMREAD_COLOR)
            if image is None or pixel_hash(image)!=src['source_pixel_sha256']:raise ValueError(f'Pixel hash mismatch: {sid}')
            stat=Path(src['source_video']).stat()
            for key,value in [('source_file_bytes',stat.st_size),('source_file_mtime_ns',stat.st_mtime_ns)]:
                if key in src and src[key]!=value:raise ValueError(f'Video metadata changed: {sid}')
            r=dict(geom['roi']);r.update(geom['new_method_roi_by_video'][src['video_id']]);roi=RoiSpec(**r)
            base=prepare_baseline(image,roi,base_config=geom['methods']['adaptive_chroma']);debug={}
            after=segment(image,roi,config=cfg,base_config=geom['methods']['adaptive_chroma'],baseline=base,debug=debug)
            before=debug['round2']
            if 'round2_mask_path' in src:
                saved=cv2.imdecode(np.fromfile(manifest.parent/src['round2_mask_path'],np.uint8),0)>0
                if not np.array_equal(saved,before.mask):raise ValueError(f'R2 baseline changed: {sid}')
            if (base.mask & ~after.mask).any():raise ValueError(f'R1 body removed: {sid}')
            if not np.array_equal(before.excluded_food,after.excluded_food):raise ValueError(f'Food changed: {sid}')
            out=a.output/'samples'/sid;out.mkdir(parents=True)
            write_image(out/'original.png',image)
            record={k:src.get(k) for k in ['sample_id','source_video','video_id','frame_index','section','group','source_pixel_sha256']}
            record.update({'evaluation_role':'frozen_transfer_same_videos' if 'round3_frozen_transfer' in str(src.get('evaluation_role')) else 'development_seen',
                'image_path':f'samples/{sid}/original.png','roi':r,
                'round2_area_px':int(before.mask.sum()),'round3_area_px':int(after.mask.sum()),
                'added_px':int((after.mask&~before.mask).sum()),'removed_px':int((before.mask&~after.mask).sum()),
                'round2_replay_bit_equal':True if 'round2_mask_path' in src else None,
                'round1_body_preserved':True,'food_unchanged':True,'accuracy_metrics':None,
                'qc_flags':after.qc_flags,'diagnostics':after.diagnostics})
            for name,result in [('round2',before),('round3',after)]:
                save_method_images(out/name,image,result)
                (out/name/'qc.json').write_text(json.dumps({'qc_flags':result.qc_flags,'diagnostics':result.diagnostics},indent=2),encoding='utf-8')
                record[name+'_mask_path']=f'samples/{sid}/{name}/mask.png'
            withheld=np.zeros(image.shape[:2],np.uint8);x0,y0,x1,y1=debug['crop_xyxy']
            withheld[y0:y1,x0:x1]=debug['withheld_crop'].astype(np.uint8)*255
            write_image(out/'withheld_review.png',withheld)
            title=f'{Path(src["source_video"]).name} | 影格 {src["frame_index"]}'
            comparison(out/'comparison.jpg',image,before,after,roi,title)
            crop=src.get('review_crop_xyxy')
            if crop:
                record['review_crop_xyxy']=crop
                comparison(out/'detail.jpg',image,before,after,roi,title+' | 固定位置放大',crop)
            record['elapsed_seconds']=time.time()-t
            (out/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            rows.append(record);print(sid,'added',record['added_px'],'removed',record['removed_px'],'seconds',round(time.time()-t,1),flush=True)
    if a.ids and set(a.ids)!=seen:raise ValueError(f'Missing requested samples: {set(a.ids)-seen}')
    if not rows:raise ValueError('No samples evaluated')
    changed=[name for name,digest in frozen['source_hashes'].items()
             if hashlib.sha256((source_dir/name).read_bytes()).hexdigest()!=digest]
    frozen.update({'samples':rows,'sample_count':len(rows),'elapsed_seconds':time.time()-start,
                   'status':'failed_source_changed' if changed else 'complete',
                   'code_files_changed_during_run':changed})
    (a.output/'results.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    if changed:raise RuntimeError(f'Source changed during run: {changed}')
    with (a.output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['sample_id','evaluation_role','round2_area_px','round3_area_px','added_px','removed_px','round2_replay_bit_equal','elapsed_seconds']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    cards=[]
    for row in rows:
        folder='samples/'+row['sample_id']
        links=' '.join(f'<a href="{folder}/{path}">{label}</a>' for path,label in [('original.png','原圖'),('round2/overlay.png','R2疊圖'),('round3/overlay.png','R3疊圖'),('round3/mask.png','二值遮罩'),('round3/preprocessed.png','前處理'),('round3/score.png','特徵強度'),('withheld_review.png','保留待確認區'),('round3/qc.json','QC')])
        detail=f'<img loading="lazy" src="{folder}/detail.jpg">' if row.get('review_crop_xyxy') else ''
        cards.append(f'<section><h2>{html.escape(row["sample_id"])}</h2><p>{html.escape(row["evaluation_role"])}；新增 {row["added_px"]:,}、移除 {row["removed_px"]:,} 像素。這些數字不是正確率。</p><p>{links}</p><p>QC：{html.escape(", ".join(row["qc_flags"]))}</p><img loading="lazy" src="{folder}/comparison.jpg">{detail}</section>')
    page='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>黏菌第三輪比較</title><style>body{font:18px sans-serif;margin:24px;background:#f5f7fa;color:#243746}img{width:100%;height:auto}section{background:white;padding:16px;margin:24px 0}a{margin-right:16px}</style><h1>同一影格：R2 與 R3</h1><p>綠色＝辨識邊界；紫色＝排除食物；橘色＝新增；藍色＝移除。沒有人工真值，不能將變更像素當成已修正像素，也不能由特徵強度估算準確率。未使用時間平滑。</p>'+''.join(cards)+'</html>'
    (a.output/'gallery.html').write_text(page,encoding='utf-8')
    print(json.dumps({'samples':len(rows),'seconds':round(time.time()-start,2)}),flush=True)


if __name__=='__main__':main()
