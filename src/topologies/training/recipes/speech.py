"""Validated CTC acoustic-model objective."""
import torch
import torch.nn.functional as F

from topologies.training.losses import LossOverrideConfig, compute_loss
from topologies.training.recipe_base import Recipe, register_recipe


class CTCConfig(LossOverrideConfig):
    blank_id: int = 0
    zero_infinity: bool = True
    reduction: str = "mean"

@register_recipe("ctc-asr",help="Length-aware CTC acoustic recognition.")
class CTCASRRecipe(Recipe):
    ConfigSchema=CTCConfig; ComponentsSchema=None; level="loss"
    def expected_batch_spec(self): return {"features":"FloatTensor[B,C,T]","lengths":"LongTensor[B]","targets":"LongTensor[N]","target_lengths":"LongTensor[B]"}
    def dummy_batch(self,device="cpu"): return {"features":torch.randn(2,1,64,device=device),"lengths":torch.tensor([64,56],device=device),"targets":torch.tensor([1,2],device=device),"target_lengths":torch.tensor([1,1],device=device)}
    def batch_to_loss(self,model,batch,cfg,components,adapter=None):
        x,lens,targ,tl=batch["features"],batch["lengths"],batch["targets"],batch["target_lengths"]
        if lens.dtype not in (torch.int32,torch.int64) or tl.dtype not in (torch.int32,torch.int64): raise ValueError("CTC lengths must be integer tensors")
        if targ.ndim!=1 or lens.shape!=(x.shape[0],) or tl.shape!=(x.shape[0],) or int(tl.sum())!=targ.numel(): raise ValueError("invalid CTC lengths/targets")
        logits,outlens=model(x,lens)
        if not 0 <= cfg.blank_id < logits.shape[-1]: raise ValueError("blank id outside model vocabulary")
        if getattr(model,"blank_id",cfg.blank_id) != cfg.blank_id: raise ValueError("recipe blank id must match model blank id")
        if targ.numel() and (torch.any(targ<0) or torch.any(targ>=logits.shape[-1]) or torch.any(targ==cfg.blank_id)): raise ValueError("CTC targets must be in-vocabulary and exclude blank")
        if torch.any(tl<0) or torch.any(lens<=0) or torch.any(lens>x.shape[-1]): raise ValueError("invalid CTC input/target lengths")
        if torch.any(tl>outlens): raise ValueError("target longer than acoustic sequence")
        if cfg.reduction not in ("none","sum","mean"): raise ValueError("invalid CTC reduction")
        if cfg.loss is None:
            loss=F.ctc_loss(logits.log_softmax(-1).transpose(0,1),targ,outlens,tl,blank=cfg.blank_id,zero_infinity=cfg.zero_infinity,reduction=cfg.reduction)
        else:
            loss=compute_loss(cfg,logits.transpose(0,1),targ,input_lengths=outlens,target_lengths=tl,blank=cfg.blank_id,zero_infinity=cfg.zero_infinity,log_probs=False,reduction=cfg.reduction)
        if loss.ndim: loss=loss.mean()
        return loss,{"loss":float(loss.detach())}
