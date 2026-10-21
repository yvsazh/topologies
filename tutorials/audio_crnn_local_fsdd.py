"""Train AudioCRNN on local FSDD WAV files (no download).

Usage: python tutorials/audio_crnn_local_fsdd.py /path/to/recordings
Recordings are grouped by the digit prefix in the standard FSDD filenames.
"""
import argparse
import wave
from pathlib import Path
import torch
from topologies.models.audio_crnn import AudioCRNN

def read_wav(path, length=8000):
    with wave.open(str(path), "rb") as wav:
        rate, channels, width, count = wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()
        if width != 2: raise ValueError(f"{path}: expected PCM16 WAV")
        data=torch.frombuffer(bytearray(wav.readframes(count)),dtype=torch.int16).float()/32768
        data=data.reshape(-1,channels).mean(1)
    if rate != 8000: raise ValueError(f"{path}: expected 8000 Hz, got {rate}; resample locally before training")
    if len(data)>length: data=data[:length]
    else: data=torch.nn.functional.pad(data,(0,length-len(data)))
    return data

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("recordings",type=Path); args=parser.parse_args()
    files=sorted(args.recordings.glob("*.wav"))
    if not files: raise SystemExit("no local WAV files found")
    examples=[]
    for f in files:
        try: label=int(f.name.split("_")[0])
        except ValueError: continue
        if 0<=label<10: examples.append((read_wav(f),label))
    if not examples: raise SystemExit("no files with FSDD digit-name convention")
    model=AudioCRNN.preset("tiny",num_classes=10,input_mode="waveform",sample_rate=8000,n_fft=256,hop_length=80)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(3):
        total=0.
        for x,y in examples:
            opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x[None],sample_rate=8000),torch.tensor([y])); loss.backward(); opt.step(); total+=float(loss.detach())
        print(f"epoch {epoch+1}: loss={total/len(examples):.4f}")
if __name__=="__main__": main()
