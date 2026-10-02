"""Export a validated user checkpoint into a chunked float32 ONNX browser model.
Run from repository root: python -m backend.export_browser_model --checkpoint PATH --samples DIR
No quantization or retraining is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from PIL import Image
from backend.service import ModelRunner, preprocess

parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--samples',required=True)
parser.add_argument('--output',default='dist/model')
args=parser.parse_args()
os.environ['GRNET_CHECKPOINT']=str(Path(args.checkpoint).resolve())
torch.set_num_threads(2)
runner=ModelRunner();status=runner.status()
if not status['ready']:raise RuntimeError(status['reason'])
model=runner.model.float().eval();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
samples=Path(args.samples)
image=np.asarray(Image.open(samples/'1.tif')).copy()
ndvi=np.asarray(Image.open(samples/'1_ndvi.tif')).copy()
data=preprocess(image,ndvi);tensor=torch.from_numpy(data[None])
started=time.perf_counter()
with torch.inference_mode():reference=model(tensor)
print('Original inference seconds',time.perf_counter()-started,flush=True)
path=out/'grnet.onnx'
torch.onnx.export(model,tensor,str(path),input_names=['image'],output_names=['green','roof'],opset_version=17,dynamo=False,external_data=False)
onnx.checker.check_model(str(path))
options=ort.SessionOptions();options.intra_op_num_threads=2
session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
outputs=session.run(None,{'image':data[None]})
validation=[]
for name,expected,actual in zip(['green','roof'],reference,outputs):
 expected=expected.detach().numpy();diff=np.abs(expected-actual)
 record={'output':name,'max_abs_error':float(diff.max()),'mean_abs_error':float(diff.mean()),'threshold_mask_disagreements':int(np.count_nonzero((expected>.1)!=(actual>.1)))}
 validation.append(record)
 if diff.max()>1e-3 or record['threshold_mask_disagreements']>90:raise ValueError('Converted model did not match source: '+str(record))
 print(record,flush=True)
green=(outputs[0][0,0]>.1)&(outputs[1][0,0]>.1);roof=outputs[1][0,0]>.1
truth_green=np.asarray(Image.open(samples/'1_mask.tif'))>0
truth_roof=np.asarray(Image.open(samples/'1_building_mask.tif'))>0
def metrics(pred,truth):
 tp=int(np.count_nonzero(pred&truth));fp=int(np.count_nonzero(pred&~truth));fn=int(np.count_nonzero(~pred&truth))
 return {'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'iou':tp/(tp+fp+fn) if tp+fp+fn else None}
result={'sample':'1','greenPixels':int(green.sum()),'roofPixels':int(roof.sum()),'roofCoverage':float(green.sum()/roof.sum()*100),'greenMetrics':metrics(green,truth_green),'buildingMetrics':metrics(roof,truth_roof)}
print('Sample inference',json.dumps(result),flush=True)
Image.fromarray((green*255).astype(np.uint8)).save(samples/'1_prediction_green.png')
Image.fromarray((roof*255).astype(np.uint8)).save(samples/'1_prediction_roof.png')
np.savez_compressed(samples/'browser-reference.npz',input=data[None],green=outputs[0],roof=outputs[1])
blob=path.read_bytes();parts=[]
for i,start in enumerate(range(0,len(blob),8*1024*1024)):
 chunk=blob[start:start+8*1024*1024];filename=f'grnet-{i:02d}.bin';(out/filename).write_bytes(chunk)
 parts.append({'file':filename,'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
manifest={'name':'用户 GR-Net 训练模型','format':'ONNX float32','checkpoint_sha256':status['checkpoint_sha256'],'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob),'parts':parts,'input':'image','outputs':['green','roof'],'input_shape':[1,5,300,300],'tile_size':300,'mask_threshold':.1,'building_gate_threshold':.1,'normalization':'BGRN /255; NDVI (value+1)/2','validation':validation,'sample_validation':result,'notes':'此指标仅为单个测试样例核验，不能代表整个测试集精度。'}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
path.unlink();print('Created',len(parts),'model chunks, bytes',len(blob),flush=True)
