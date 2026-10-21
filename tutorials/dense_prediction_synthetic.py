"""Offline synthetic shapes (segmentation) and rectangles (object detection)."""
import torch
from topologies.models.unet import UNet
from topologies.models.retinanet import RetinaNet
from topologies.training.recipes.segmentation import SegmentationRecipe, SegmentationConfig
from topologies.training.recipes.detection import DetectionRecipe, DetectionConfig

def shapes(n=8,size=32):
    x=torch.zeros(n,1,size,size); y=torch.zeros(n,size,size,dtype=torch.long)
    for i in range(n):
        a=4+i%5; b=8+i%4; x[i,0,a:a+8,b:b+7]=1; y[i,a:a+8,b:b+7]=1
    return {"pixels":x,"mask":y}

def train_once():
    seg=UNet(1,2,(8,16)); sr=SegmentationRecipe(); sb=shapes(); loss,_=sr.batch_to_loss(seg,sb,SegmentationConfig(),None); loss.backward()
    dcfg=DetectionConfig(num_classes=1,anchor_sizes=(8.,16.,32.))
    det=RetinaNet(3,1,16,anchor_sizes=dcfg.anchor_sizes); dr=DetectionRecipe(); db=dr.dummy_batch(); dl,_=dr.batch_to_loss(det,db,dcfg,None); dl.backward()
    det.eval()
    with torch.no_grad(): detections=dr.predict_step(det,db,dcfg.model_copy(update={"score_threshold":0.0}),None)
    # Synthetic local evidence only; deliberately no COCO/mAP score claim.
    print({"segmentation_loss":float(loss.detach()),"detection_loss":float(dl.detach()),"detections_per_image":[len(d["scores"]) for d in detections]})

if __name__=="__main__": train_once()
