# ChordMax — Automatic Chord Recognition

Sistema de reconhecimento automático de acordes musicais a partir de áudio, baseado em arquiteturas Transformer (BTC) e Conformer (ChordFormer) com decomposição estrutural de acordes em 9 componentes independentes.

## Arquitetura

```
Audio (.mp3/.wav)
    → CQT (252 bins, 36/oitava)
    → ConformerEncoder (12 blocos, 4 heads, hidden=128)
    → 9 cabeças de classificação independentes
    → Reassemble → Símbolo de acorde (ex: C:maj7, A:min9/E)
```

### 9 Componentes

| Componente | Classes | Exemplo |
|---|---|---|
| Root | 13 | C, D, E, ... |
| Bass | 13 | N, E, G, ... |
| Triad | 7 | maj, min, dim, aug, sus2, sus4 |
| Misc | 2 | N, 5 (power chord) |
| 6th | 2 | N, 6 |
| 7th | 4 | N, 7, b7, bb7 |
| 9th | 4 | N, 9, #9, b9 |
| 11th | 3 | N, 11, #11 |
| 13th | 3 | N, 13, b13 |

**Total: 51 classes** (vs 170 monolíticas)

## Uso Rápido

### Treino (ChordFormer + GradNorm)

```bash
cd BTC-ISMIR19
python train_decomposed.py \
    --config run_config.yaml \
    --backbone chordformer \
    --kfold 0 \
    --use_gradnorm \
    --wandb_project chordMax
```

### Inferência

```bash
python infer_full_audio.py \
    --config run_config.yaml \
    --checkpoint checkpoints/model_best.pt \
    --audio_file musica.mp3 \
    --output resultado.lab
```

## Documentação

- [Pipeline Completo](BTC-ISMIR19/docs/PIPELINE_DOCUMENTATION.md)
- [Guia de Decomposição](BTC-ISMIR19/docs/CHORD_DECOMPOSITION_GUIDE.md)
- [Guia de Treinamento e Inferência](BTC-ISMIR19/docs/GUIA_TREINAMENTO_E_INFERENCIA.md)

## Requisitos

- Python >= 3.10
- PyTorch >= 2.0
- librosa >= 0.10
- mir_eval >= 0.7
- wandb >= 0.22 (opcional)

```bash
pip install -r BTC-ISMIR19/requirements.txt
```
