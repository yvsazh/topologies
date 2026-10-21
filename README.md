# topologies

A PyTorch toolbox for building, training, comparing and inspecting neural-network
architectures — plus a catalog of small reference implementations (GPT, LLaMA, ViT,
ResNet, BERT, Mamba, ...).

> **Early alpha (v0.0.1).** The toolbox — training engine, data utilities, tokenizers,
> model constructor, generation, evaluation, experiments, introspection — is tested.
> Catalog models in `topologies.models` / `topologies.layers` are experimental
> references: importable, trainable and smoke-tested, but not validated against the papers.
> APIs may change between releases.

## Install

```bash
pip install topologies                 # core
pip install "topologies[all]"          # + vision, nlp, tensorboard, visualization, ...
```

Python 3.10–3.13, PyTorch `>=2.2,<2.13`.

## Quickstart

Train a tiny language model on any text corpus:

```python
from topologies import Trainer, build, data, generate, train_tokenizer
from topologies.nn import TransformerLM

docs = data.read_documents("corpus.txt")
train_docs, val_docs = data.train_val_split(docs, 0.1)
tok = train_tokenizer(train_docs, kind="bpe", vocab_size=2048)
model = build(TransformerLM, "tiny", vocab_size=tok.vocab_size, max_seq=128)

trainer = Trainer(model, data.lm_dataset(train_docs, tok, 128),
                  data.lm_dataset(val_docs, tok, 128),
                  tokenizer=tok, batch_size=32, max_epochs=20, lr=1e-3)
trainer.fit()
trainer.save_pretrained("artifacts/tiny-lm")
print(generate(model, "Once upon a time", tokenizer=tok, max_new_tokens=100, top_k=40))
```

Compare many configurations with one declaration — the library runs every trial
(resumably), then ranks and explains the results:

```python
from topologies import Experiment
from topologies.experiments import LanguageModeling, Variant

exp = Experiment("arch-x-tokenizer",
                 task=LanguageModeling(docs, val_fraction=0.1, block_size=128),
                 axes={"model": [TransformerLM,
                                 Variant("llama-ish", TransformerLM, pos="rope",
                                         norm="rms", activation="swiglu")],
                       "tokenizer": ["char", "bpe:1k", "bpe:4k"]},
                 target_params="1M", train_for="2 epochs", seeds=2)
exp.run().show()
```

## What's inside

- `topologies.nn` — composable stacks, presets and parameter budgets
- `Trainer`, `generate`, evaluation, optimization and quantization helpers
- `topologies.data`, dataset catalog, tokenizers
- `Experiment` — sweeps, leaderboards, pivots, importance analysis
- `topologies.introspection` / `topologies.visualization` — look inside trained models
- `topologies` CLI — `topologies --help`

## Links

- Documentation: <https://topologies.hackwhiz.net>
- Repository and issues: <https://github.com/yvsazh/topologies>

## License

Apache-2.0
