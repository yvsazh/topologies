"""No-download CTC acoustic alignment demonstration with synthetic tones."""
import torch
import torch.nn.functional as F
from topologies.models.ctc_acoustic_model import CTCAcousticModel, greedy_ctc_collapse

torch.manual_seed(11)
model=CTCAcousticModel(vocab_size=4,conv_channels=8,hidden_size=8,rnn_layers=1)
features=torch.randn(2,1,160)
input_lengths=torch.tensor([160,128])
targets=torch.tensor([1,2,3]); target_lengths=torch.tensor([2,1])
logits,out_lengths=model(features,input_lengths)
loss=F.ctc_loss(logits.log_softmax(-1).transpose(0,1),targets,out_lengths,target_lengths,blank=0,zero_infinity=True)
loss.backward()
print("CTC loss",float(loss.detach()),"greedy",greedy_ctc_collapse(logits,out_lengths,blank=0))
