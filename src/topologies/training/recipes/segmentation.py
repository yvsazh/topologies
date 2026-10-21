"""Pixel-wise semantic segmentation training with CE/BCE and optional Dice."""
import torch
import torch.nn.functional as F
from pydantic import Field

from topologies.evaluation import EvaluationRecord
from topologies.training.losses import LossOverrideConfig, compute_loss
from topologies.training.recipe_base import EvaluationStepResult, Recipe, register_recipe


class SegmentationConfig(LossOverrideConfig):
    num_classes: int = Field(default=2, gt=0)
    ignore_index: int | None = -100
    dice_weight: float = Field(default=0.0, ge=0.0)
    binary: bool = False

@register_recipe("segmentation", help="Pixel-level semantic segmentation with CE/BCE and optional Dice.")
class SegmentationRecipe(Recipe):
    level="loss"; ConfigSchema=SegmentationConfig; ComponentsSchema=None
    def expected_batch_spec(self): return {"pixels":"FloatTensor[B,C,H,W]", "mask":"LongTensor[B,H,W] (or float mask for binary)"}
    def dummy_batch(self, device="cpu"): return {"pixels":torch.randn(2,1,16,20,device=device),"mask":torch.randint(0,2,(2,16,20),device=device)}
    def default_training_policy(self): return {"optimizer":"adamw","lr":1e-3,"weight_decay":0.0,"scheduler":None}
    def validate_batch(self,batch,*,strict=True):
        super().validate_batch(batch,strict=strict)
        if strict:
            pixels,mask=batch["pixels"],batch["mask"]
            if pixels.ndim!=4 or mask.ndim!=3 or pixels.shape[0]!=mask.shape[0] or pixels.shape[-2:]!=mask.shape[-2:]:
                raise ValueError("segmentation batch requires aligned pixels [B,C,H,W] and mask [B,H,W]")
    def preflight(self,model,batch,cfg,components,adapter=None):
        self.validate_batch(batch,strict=True); logits=self._forward(model,batch)
        expected=1 if cfg.binary else cfg.num_classes
        if logits.shape[1]!=expected: raise ValueError(f"segmentation model emits {logits.shape[1]} channels; expected {expected}")
        mask=batch["mask"]
        if not cfg.binary:
            valid=mask.ne(cfg.ignore_index) if cfg.ignore_index is not None else torch.ones_like(mask,dtype=torch.bool)
            if valid.any() and (mask[valid].min()<0 or mask[valid].max()>=cfg.num_classes): raise ValueError("segmentation target class outside configured class range")
    def _forward(self, model, batch):
        logits=model(batch["pixels"])
        if logits.ndim!=4 or logits.shape[0]!=batch["pixels"].shape[0] or logits.shape[-2:]!=batch["pixels"].shape[-2:]: raise ValueError("segmentation model must return [B,K,H,W] aligned logits")
        return logits
    def _loss(self, logits, mask, cfg):
        valid=mask.ne(cfg.ignore_index) if cfg.ignore_index is not None else torch.ones_like(mask,dtype=torch.bool)
        if cfg.loss is not None:
            target = mask.float().unsqueeze(1) if cfg.binary else mask.long()
            return compute_loss(cfg, logits, target, mask=valid), valid
        if cfg.binary:
            target=mask.float(); target=target.unsqueeze(1) if target.ndim==3 else target
            per=F.binary_cross_entropy_with_logits(logits,target,reduction="none"); vm=valid.unsqueeze(1).expand_as(per)
            loss=per[vm].mean() if vm.any() else logits.sum()*0
            probs=logits.sigmoid(); dice=1-(2*(probs*target*vm).sum()+1)/((probs*vm).sum()+(target*vm).sum()+1)
        else:
            loss=(F.cross_entropy(logits,mask.long(),ignore_index=cfg.ignore_index if cfg.ignore_index is not None else -100)
                  if valid.any() else logits.sum()*0)
            probs=logits.softmax(1); safe=mask.masked_fill(~valid,0).long(); truth=F.one_hot(safe,num_classes=logits.shape[1]).permute(0,3,1,2).to(probs.dtype); vm=valid[:,None]
            dims=(0,2,3); dice=1-((2*(probs*truth*vm).sum(dims)+1)/((probs*vm).sum(dims)+(truth*vm).sum(dims)+1)).mean()
        return loss+cfg.dice_weight*dice,valid
    def batch_to_loss(self,model,batch,cfg,components,adapter=None):
        logits=self._forward(model,batch); loss,valid=self._loss(logits,batch["mask"],cfg); pred=logits.sigmoid().ge(.5).squeeze(1) if cfg.binary else logits.argmax(1)
        metric=((pred==batch["mask"])&valid).sum()/valid.sum().clamp_min(1)
        return loss,{"loss":float(loss.detach()),"pixel_accuracy":float(metric.detach())}
    def predict_step(self,model,batch,cfg,components,adapter=None):
        logits=self._forward(model,batch); return logits.sigmoid().ge(.5) if cfg.binary else logits.argmax(1)
    def evaluation_step(self,model,batch,cfg,components,adapter=None):
        logits=self._forward(model,batch); loss,valid=self._loss(logits,batch["mask"],cfg); pred=logits.sigmoid().ge(.5).squeeze(1) if cfg.binary else logits.argmax(1); target=batch["mask"]
        correct=(((pred==target)&valid).sum()).float(); acc=correct/valid.sum().clamp_min(1)
        if cfg.binary:
            intersection=((pred & target.bool() & valid).sum()).float(); union=((pred & valid).sum()+(target.bool() & valid).sum()-intersection).float()
        else:
            classes=torch.arange(logits.shape[1],device=logits.device)
            intersections=torch.stack([((pred==c)&(target==c)&valid).sum() for c in classes]).float()
            unions=torch.stack([(((pred==c)|(target==c))&valid).sum() for c in classes]).float()
            present=unions>0; intersections=intersections[1:]; unions=unions[1:]; present=present[1:]
            miou=(intersections[present]/unions[present]).mean() if present.any() else logits.sum()*0
            return EvaluationStepResult({"loss":float(loss.detach()),"pixel_accuracy":float(acc),"mean_foreground_iou":float(miou)},EvaluationRecord(pred,target,metadata=batch))
        iou=intersection/union.clamp_min(1)
        return EvaluationStepResult({"loss":float(loss.detach()),"pixel_accuracy":float(acc),"foreground_iou":float(iou)},EvaluationRecord(pred,target,metadata=batch))
