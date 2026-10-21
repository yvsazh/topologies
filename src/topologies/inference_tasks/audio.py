"""Pure helpers for audio inference that do not alter model behavior."""
import torch


def greedy_ctc_decode(logits, lengths, blank_id=0):
    """Argmax decode with CTC repeat merging and blank removal."""
    if logits.ndim != 3 or lengths.shape != (logits.shape[0],):
        raise ValueError("logits must be [B,T,V], lengths [B]")
    if not 0 <= blank_id < logits.shape[-1] or torch.any(lengths < 0) or torch.any(lengths > logits.shape[1]):
        raise ValueError("invalid blank id or input lengths")
    decoded=[]
    for row,length in zip(logits.argmax(-1),lengths):
        ids=row[:int(length)].tolist(); result=[]; previous=None
        for token in ids:
            if token != previous and token != blank_id: result.append(token)
            previous=token
        decoded.append(result)
    return decoded

def autoregressive_sample(model, prefix, steps, *, conditioning=None, temperature=1.0):
    if steps < 0 or temperature <= 0: raise ValueError("steps >= 0 and temperature > 0 required")
    tokens=prefix
    for _ in range(steps):
        probs=(model(tokens,conditioning)[:,-1]/temperature).softmax(-1)
        tokens=torch.cat((tokens,torch.multinomial(probs,1)),dim=1)
    return tokens
