"""Narrow audio objectives: spectral reconstruction and shifted waveform CE."""
import torch
import torch.nn.functional as F

from topologies.evaluation import EvaluationRecord
from topologies.training.losses import LossOverrideConfig, compute_loss
from topologies.training.recipe_base import (
    EvaluationStepResult,
    Recipe,
    RecipeConfig,
    register_recipe,
)


class AudioConfig(LossOverrideConfig):
    n_fft: int = 128
    hop_length: int = 32
    mode: str = "ddsp-controls"
    sample_rate: int = 16000
    resolutions: tuple[tuple[int, int], ...] = ((128, 32), (64, 16))


@register_recipe("audio-classify", help="Waveform classification with cross-entropy.")
class AudioClassifyRecipe(Recipe):
    ConfigSchema = RecipeConfig
    ComponentsSchema = None
    level = "loss"

    def expected_batch_spec(self):
        return {"x": "FloatTensor[B,T]", "labels": "LongTensor[B]"}

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        logits = model(batch["x"])
        loss = F.cross_entropy(logits, batch["labels"].long())
        return loss, {"loss": float(loss.detach()), "accuracy": float((logits.argmax(-1) == batch["labels"]).float().mean())}

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        logits = model(batch["x"])
        labels = batch["labels"].long()
        loss = F.cross_entropy(logits, labels)
        return EvaluationStepResult({"loss": float(loss), "accuracy": float((logits.argmax(-1) == labels).float().mean())}, EvaluationRecord(logits, labels, metadata=batch))

@register_recipe("audio-reconstruct", help="Multi-term log-magnitude STFT reconstruction.")
class AudioReconstructRecipe(Recipe):
    ConfigSchema=AudioConfig; ComponentsSchema=None; level="loss"
    def expected_batch_spec(self):
        return {"audio":"FloatTensor[B,T]", "pitch":"FloatTensor[B,K]", "loudness":"FloatTensor[B,K]", "sample_rate":"int", "lengths":"LongTensor[B] optional"}
    def optional_batch_keys(self): return {"lengths"}
    def dummy_batch(self,device="cpu"):
        return {"audio":torch.zeros(2,512,device=device),"pitch":torch.full((2,4),220.,device=device),"loudness":torch.ones(2,4,device=device),"sample_rate":16000,"lengths":torch.full((2,),512,device=device,dtype=torch.long)}
    def batch_to_loss(self,model,batch,cfg,components,adapter=None):
        x=batch["audio"]
        if cfg.mode == "ddsp-controls":
            if int(batch["sample_rate"]) != cfg.sample_rate or cfg.sample_rate != model.sample_rate: raise ValueError("audio/model sample rate mismatch")
            output=model(batch["pitch"],batch["loudness"],num_samples=x.shape[-1])
            if not isinstance(output,tuple) or len(output)!=2: raise TypeError("DDSP must return (waveform, phase_state)")
            y,_state=output
        elif cfg.mode == "wave-autoencoder":
            y=model(x)
        else: raise ValueError("mode must be ddsp-controls or wave-autoencoder")
        if y.shape != x.shape: raise ValueError("reconstruction waveform shape must match target")
        lengths=batch.get("lengths",torch.full((x.shape[0],),x.shape[-1],device=x.device,dtype=torch.long))
        if lengths.shape!=(x.shape[0],) or torch.any(lengths<=0) or torch.any(lengths>x.shape[-1]): raise ValueError("invalid valid waveform lengths")
        if cfg.loss is None:
            win=torch.hann_window(cfg.n_fft,device=x.device)
            a=torch.stft(y,cfg.n_fft,cfg.hop_length,window=win,return_complex=True).abs()
            b=torch.stft(x,cfg.n_fft,cfg.hop_length,window=win,return_complex=True).abs()
            frame_lengths=lengths//cfg.hop_length+1
            frame_mask=torch.arange(a.shape[-1],device=x.device)[None,:] < frame_lengths[:,None]
            loss=F.l1_loss(torch.log1p(a)*frame_mask[:,None,:],torch.log1p(b)*frame_mask[:,None,:])
        else:
            valid = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
            if cfg.loss == "multi-resolution-stft":
                loss=compute_loss(cfg,y*valid,x*valid,resolutions=cfg.resolutions)
            else:
                loss=compute_loss(cfg,y*valid,x*valid)
        return loss,{"loss":float(loss.detach())}

@register_recipe("waveform-lm",help="Shifted categorical waveform language modeling.")
class WaveformLMRecipe(Recipe):
    ConfigSchema=RecipeConfig; ComponentsSchema=None; level="loss"
    def expected_batch_spec(self): return {"tokens":"LongTensor[B,T]"}
    def dummy_batch(self,device="cpu"): return {"tokens":torch.randint(0,32,(2,32),device=device)}
    def batch_to_loss(self,model,batch,cfg,components,adapter=None):
        t=batch["tokens"]
        if t.ndim!=2 or t.shape[1]<2: raise ValueError("tokens must be [B,T>=2]")
        logits=model(t[:,:-1]); loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),t[:,1:].reshape(-1))
        return loss,{"loss":float(loss.detach())}
