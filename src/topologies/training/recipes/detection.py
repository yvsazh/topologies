"""Anchor-based detection recipe using per-image boxes and focal/Smooth-L1 losses."""
import torch
from pydantic import Field

from topologies.inference_tasks.detection import (
    detection_loss,
    postprocess_detections,
    undo_resize_pad,
)
from topologies.models.retinanet import generate_anchors
from topologies.training.recipe_base import Recipe, RecipeComponents, RecipeConfig, register_recipe


class DetectionConfig(RecipeConfig):
    num_classes:int=Field(gt=0,default=1)
    score_threshold:float=Field(default=.05,ge=0.,le=1.)
    nms_threshold:float=Field(default=.5,ge=0.,le=1.)
    positive_iou:float=Field(default=.5,gt=0.,le=1.)
    negative_iou:float=Field(default=.4,ge=0.,lt=1.)
    max_detections:int=Field(default=100,gt=0)
    anchor_scales:tuple[float,...]=(1., 2**(1/3), 2**(2/3))
    anchor_ratios:tuple[float,...]=(.5,1.,2.)
    anchor_sizes:tuple[float,...]=(32.,64.,128.)

    @property
    def anchors_per_location(self): return len(self.anchor_scales)*len(self.anchor_ratios)

class DetectionComponents(RecipeComponents):
    """Optional typed anchor override; head count must agree with model construction."""
    anchor_scales:tuple[float,...]=(1., 2**(1/3), 2**(2/3))
    anchor_ratios:tuple[float,...]=(.5,1.,2.)
    anchor_sizes:tuple[float,...]=(32.,64.,128.)

@register_recipe("detection",help="RetinaNet anchor matching, sigmoid focal classification and Smooth-L1 box regression.")
class DetectionRecipe(Recipe):
    level="loss"; ConfigSchema=DetectionConfig; ComponentsSchema=DetectionComponents
    def expected_batch_spec(self): return {"images":"list[FloatTensor[C,H_i,W_i]] or padded FloatTensor[B,C,H,W]","targets":"list[{boxes:FloatTensor[N,4],labels:LongTensor[N]}]"}
    def optional_batch_keys(self): return {"image_sizes","scale","pad_xy"}
    def default_training_policy(self): return {"optimizer":"adamw","lr":1e-4,"weight_decay":1e-4,"scheduler":None}
    def dummy_batch(self,device="cpu"):
        images=[torch.randn(3,32,40,device=device),torch.randn(3,28,35,device=device)]
        return {"images":images,"targets":[{"boxes":torch.tensor([[8.,8.,24.,24.]],device=device),"labels":torch.tensor([0],device=device)},{"boxes":torch.empty((0,4),device=device),"labels":torch.empty((0,),dtype=torch.long,device=device)}]}
    def _images(self,batch):
        imgs=batch["images"]
        if isinstance(imgs,torch.Tensor): return imgs
        if not imgs: raise ValueError("images batch cannot be empty")
        c=imgs[0].shape[0]; h=max(x.shape[-2] for x in imgs); w=max(x.shape[-1] for x in imgs)
        out=imgs[0].new_zeros((len(imgs),c,h,w))
        for i,x in enumerate(imgs):
            if x.ndim!=3 or x.shape[0]!=c: raise ValueError("all images need [C,H,W] with equal channels")
            out[i,:,:x.shape[-2],:x.shape[-1]]=x
        return out
    def _anchors(self,grid,stride,images,cls,model,cfg,components,level):
        scales=components.anchor_scales if components is not None else cfg.anchor_scales
        ratios=components.anchor_ratios if components is not None else cfg.anchor_ratios
        sizes=components.anchor_sizes if components is not None else cfg.anchor_sizes
        expected=len(scales)*len(ratios)
        if expected!=model.anchors_per_location or cls.shape[1]!=grid.shape[0]*grid.shape[1]*expected:
            raise ValueError(f"anchor geometry has {expected} anchors/location but model head declares {model.anchors_per_location}")
        if tuple(scales)!=tuple(model.anchor_scales) or tuple(ratios)!=tuple(model.anchor_ratios) or tuple(sizes)!=tuple(model.anchor_sizes):
            raise ValueError("recipe anchor_scales/anchor_ratios must exactly match RetinaNet constructor configuration")
        return generate_anchors(grid.shape[0],grid.shape[1],stride,scales=scales,ratios=ratios,device=images.device,dtype=images.dtype,base_size=sizes[level])
    def validate_batch(self,batch,*,strict=True):
        super().validate_batch(batch,strict=strict)
        if not strict: return
        imgs=batch["images"]; targets=batch["targets"]
        n=imgs.shape[0] if isinstance(imgs,torch.Tensor) else len(imgs)
        if len(targets)!=n: raise ValueError("targets must contain one boxes/labels record per image")
        for image,target in zip(imgs,targets):
            if "boxes" not in target or "labels" not in target: raise ValueError("each target needs boxes and labels")
            boxes,labels=target["boxes"],target["labels"]
            if boxes.ndim!=2 or boxes.shape[-1]!=4 or labels.ndim!=1 or labels.shape[0]!=boxes.shape[0]: raise ValueError("target boxes [N,4] and labels [N] must align")
            if boxes.numel() and (not torch.isfinite(boxes).all() or torch.any(boxes[:,2:]<=boxes[:,:2])): raise ValueError("target boxes must be finite, nondegenerate xyxy")
            if labels.numel() and (labels.min()<0): raise ValueError("target labels are zero-based nonnegative class ids")
    def preflight(self,model,batch,cfg,components,adapter=None):
        self.validate_batch(batch,strict=True)
        if model.num_classes!=cfg.num_classes: raise ValueError("detection config num_classes must match detector head")
        images=self._images(batch); output=model(images)
        if len(output.class_logits)!=len(output.box_deltas): raise ValueError("detector level logits/deltas mismatch")
        self._anchors(output.feature_grids[0],output.strides[0],images,output.class_logits[0],model,cfg,components,0)
        for target in batch["targets"]:
            if target["labels"].numel() and target["labels"].max()>=cfg.num_classes: raise ValueError("target label exceeds configured class range")
    def _loss(self,model,batch,cfg,components=None):
        self.validate_batch(batch,strict=True)
        images=self._images(batch); output=model(images); terms=[]; count=0
        for i,target in enumerate(batch["targets"]):
            losses=[]
            for level,(cls,box,grid,stride) in enumerate(zip(output.class_logits,output.box_deltas,output.feature_grids,output.strides)):
                anchors=self._anchors(grid,stride,images,cls,model,cfg,components,level)
                if target["labels"].numel() and target["labels"].max()>=cfg.num_classes: raise ValueError("target label exceeds configured class range")
                losses.append(detection_loss(cls[i],box[i],anchors,target["boxes"].to(images),target["labels"].to(images.device),cfg.num_classes,cfg.positive_iou,cfg.negative_iou)[0])
            terms.append(torch.stack(losses).sum()); count+=1
        return torch.stack(terms).sum()/max(count,1),output,images
    def batch_to_loss(self,model,batch,cfg,components,adapter=None):
        loss,_,_=self._loss(model,batch,cfg,components); return loss,{"loss":float(loss.detach())}
    def predict_step(self,model,batch,cfg,components,adapter=None):
        images=self._images(batch); output=model(images); results=[]
        for i in range(images.shape[0]):
            rows=[]
            for level,(cls,box,grid,stride) in enumerate(zip(output.class_logits,output.box_deltas,output.feature_grids,output.strides)):
                anchors=self._anchors(grid,stride,images,cls,model,cfg,components,level)
                rows.append(postprocess_detections(cls[i],box[i],anchors,cfg.score_threshold,cfg.nms_threshold,cfg.max_detections))
            boxes=torch.cat([r["boxes"] for r in rows]); scores=torch.cat([r["scores"] for r in rows]); labels=torch.cat([r["labels"] for r in rows]); order=scores.argsort(descending=True)[:cfg.max_detections]
            result={"boxes":boxes[order],"scores":scores[order],"labels":labels[order]}
            sizes=batch.get("image_sizes")
            if sizes is None and not isinstance(batch["images"],torch.Tensor): sizes=[tuple(im.shape[-2:]) for im in batch["images"]]
            if "scale" in batch:
                scales=batch["scale"] if isinstance(batch["scale"],(list,tuple)) else [batch["scale"]]*images.shape[0]
                pads=batch.get("pad_xy",[(0,0)]*images.shape[0]); result["boxes"]=undo_resize_pad(result["boxes"],scales[i],pads[i],sizes[i] if sizes is not None else None)
            elif sizes is not None:
                result["boxes"]=undo_resize_pad(result["boxes"],1.,(0.,0.),sizes[i])
            results.append(result)
        return results


@register_recipe("object-detection", help="RetinaNet anchor matching and focal/box losses.")
class ObjectDetectionRecipe(DetectionRecipe):
    pass
