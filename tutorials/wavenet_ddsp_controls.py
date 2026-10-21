"""No-download categorical waveform and differentiable control demo."""
import torch
import torch.nn.functional as F
from topologies.models.wavenet import WaveNet, mu_law_encode
from topologies.models.ddsp_synthesizer import DDSPSynthesizer

torch.manual_seed(12)
wave=torch.sin(torch.linspace(0,8*torch.pi,512)).unsqueeze(0)
tokens=mu_law_encode(wave,32)
model=WaveNet(quantization_channels=32,residual_channels=8,layers=4)
logits=model(tokens[:,:-1]); loss=F.cross_entropy(logits.reshape(-1,32),tokens[:,1:].reshape(-1)); loss.backward()
print("WaveNet shifted CE",float(loss.detach()))
synth=DDSPSynthesizer(sample_rate=8000,n_harmonics=12,control_rate=100,fir_size=5)
pitch=torch.full((1,8),220.); loudness=torch.linspace(.1,1,8).unsqueeze(0)
audio,state=synth(pitch,loudness,num_samples=512)
print("DDSP samples",audio.shape[-1],"state",state.tolist(),"spectral bins",torch.stft(audio,128,32,return_complex=True).abs().shape[-2])
