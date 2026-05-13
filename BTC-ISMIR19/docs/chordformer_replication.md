# Replicating ChordFormer inside ChordMax for fair head-to-head benchmarks

This document explains how to run the ChordMax codebase in a configuration
that **mirrors the ChordFormer paper** (see
[`chordformer_vs_chordmax_tables.tex`](./chordformer_vs_chordmax_tables.tex))
so the two models can be compared on the same datasets, with the same
optimizer/scheduler/loss/CRF, and using our 9 output heads.

The goal here is **not** to reduce ChordMax to 6 heads or to swap our datasets
for the Humphrey-Bello collection. Those choices are intentionally kept as in
ChordMax. Only the input representation, encoder, loss, CRF, optimizer, and
scheduler are aligned to the paper.

---

## 1. What each table row maps to

Below, each row of the comparison tables in
[`chordformer_vs_chordmax_tables.tex`](./chordformer_vs_chordmax_tables.tex)
is mapped to the YAML key or CLI flag that controls it.

### Tabela 1 — Input representation

| Paper row              | ChordFormer value         | YAML key (file)                                         |
| ---------------------- | ------------------------- | ------------------------------------------------------- |
| Transformada           | CQT em dB (ref. máx.)     | `utils/preprocess.py` (unchanged — CQT log-magnitude)    |
| Taxa de amostragem     | 22 050 Hz                 | `mp3.song_hz: 22050` (`run_config_chordformer.yaml`)    |
| Bins de frequência     | 252                       | `feature.n_bins: 252`                                   |
| Bins por oitava        | 36                        | `feature.bins_per_octave: 36`                           |
| Oitavas                | C1 – C8                   | derivado de `n_bins` / `bins_per_octave`                |
| **Hop length**         | **512 amostras (~23 ms)** | `feature.hop_length: 512` (ChordMax usa 2048)           |
| **Janela de entrada**  | **1 000 frames (~23,2 s)**| `mp3.inst_len: 23.22` + `model.timestep: 1000`          |

### Tabela 2 — Conformer encoder

| Paper row                | ChordFormer value          | YAML key / flag                                            |
| ------------------------ | -------------------------- | ---------------------------------------------------------- |
| Projeção de entrada      | Linear 252 → 256           | `model.feature_size: 252`, `model.hidden_size: 256`        |
| Dimensão oculta          | 256                        | `model.hidden_size: 256`                                   |
| N.º de blocos            | 4                          | `model.num_layers: 4`                                      |
| Heads de atenção         | 16                         | `model.num_heads: 16`                                      |
| Dim. head (`d_k`)        | 16                         | derivado (`hidden_size // num_heads`)                      |
| Dimensão FFN             | 1 024 (4×)                 | `model.ff_expansion_factor: 4` (4 × 256 = 1 024)           |
| Kernel depthwise conv    | 31                         | `model.conv_kernel_size: 31`                               |
| Expansão conv            | não especificado           | `model.conv_expansion_factor: 2` (ChordMax default mantido)|
| **Codif. posicional**    | **Nenhuma**                | `model.use_positional_encoding: False` (novo flag)         |
| Normalização             | LayerNorm (pre-norm)       | hard-coded em `ConformerBlock`                             |
| Ativação FFN             | Swish                      | hard-coded em `ConformerFeedForward`                       |
| Batch Norm (conv)        | Sim                        | `model.use_batchnorm_in_conv: True` (novo flag)            |
| Parâmetros totais        | não reportado              | reportado no log inicial do `train_decomposed.py`          |

### Tabela 4 — Heads de saída (mantidas iguais às do ChordMax)

A variante ChordFormer-like **mantém as 9 heads do ChordMax**
(Root/Bass/Triad/Misc/6th/7th/9th/11th/13th = 51 classes totais)
para não perder informação de vocabulário. Esta é uma decisão explícita
da comparação: queremos isolar o efeito da arquitetura/CRF/treino, e não
do espaço de saída.

### Tabela 5 — Loss, balanceamento e decodificação

| Paper row              | ChordFormer value                  | YAML key / flag                                              |
| ---------------------- | ---------------------------------- | ------------------------------------------------------------ |
| Perda base             | Cross-entropy ponderada            | `class_weights.enabled: True`                                |
| `gamma`                | 0,3 / 0,5 / 0,7 / 1,0 (testados)   | `class_weights.gamma: 0.5` (sweepable)                       |
| `w_max`                | 10,0 ou 20,0                       | `class_weights.w_max: 10.0`                                  |
| GradNorm dinâmico      | **Não**                            | `gradnorm.enabled: False` ou `--disable_gradnorm`            |
| Focal loss             | **Não**                            | `focal.gamma: 0.0`                                           |
| **CRF**                | **Linear, lambda = 30 (fixo)**     | `crf.type: linear`, `crf.lambda: 30.0` (ou `--crf linear --crf_lambda 30`) |
| Modo CRF               | Único (6 heads no paper)           | `--crf_mode root_triad` (91 tags) ou `full` em `train_harmonic_crf.py` |

A classe `LinearCRF` (em
[`models/linear_crf.py`](../models/linear_crf.py)) implementa a matriz de
transição fixa `T = lambda * I` (não-treinável) e expõe a mesma API do `CRF`
treinável existente; é selecionável tanto em `train_harmonic_crf.py`
(`--crf_kind linear`) quanto persistida no checkpoint do backbone via
`train_decomposed.py` (`--crf linear`).

### Tabela 6 — Hiperparâmetros de treinamento

| Paper row              | ChordFormer value                       | YAML key / flag                                           |
| ---------------------- | --------------------------------------- | --------------------------------------------------------- |
| **Otimizador**         | **AdamW**                               | `experiment.optimizer: adamw` ou `--optimizer adamw`      |
| Learning rate          | 1 × 10⁻³                                | `--learning_rate 1e-3`                                    |
| **Scheduler**          | **ReduceLROnPlateau (÷10 após 5 ep.)**  | `experiment.scheduler: plateau`, `scheduler_factor: 0.1`, `scheduler_patience: 5` |
| LR mínimo              | 1 × 10⁻⁶                                | `experiment.scheduler_min_lr: 1.0e-6` (early-stop por LR) |
| Weight decay           | AdamW (implícito)                       | `--weight_decay 0.01`                                     |
| Batch size             | 24 seg. × 1 000 frames                  | `--batch_size 24` (e `model.timestep: 1000`)              |
| Épocas máx.            | Até LR ≤ 10⁻⁶                          | `--num_epochs 200` + early-stop por LR                    |
| Validação              | 5-fold (60/20/20 %)                     | `--kfold 0..4` (split atual: 80/20 por fold)              |

> **Nota sobre validação:** ChordMax atualmente usa hold-one-fold (≈80/20 por
> fold). Migrar para o split 60/20/20 (train/val/test separados) está fora do
> escopo desta comparação — para um benchmark justo, basta rodar os dois
> modelos no mesmo split atual.

### Tabela 7 — Dataset e augmentation

| Paper row       | ChordFormer value                         | ChordMax (esta replicação)                                   |
| --------------- | ----------------------------------------- | ------------------------------------------------------------ |
| Dataset         | Humphrey-Bello (1217 músicas)             | Billboard + Djavan SB1·SB2 + JAAH + Queen + RW + RWC         |
| Foco musical    | Pop/Rock ocidental                        | MPB + Jazz + Pop                                             |
| Pitch shifting  | −5 a +6 semitons (CQT)                    | Idem                                                         |

Mantemos o nosso corpus por uma razão simples: queremos comparar
**arquitetura ChordFormer vs arquitetura ChordMax no nosso domínio musical**.
Trocar o dataset misturaria duas variáveis ao mesmo tempo.

---

## 2. Como reproduzir o experimento (passo a passo)

### Passo 1 — Pré-processar com hop = 512

Os `.pt` existentes (gerados com `hop_length=2048`) **não servem** para esta
variante. Rode o pré-processamento apontando para o YAML novo. Como o
caminho de saída inclui `feature_string = "cqt_252_36_512"`, os novos `.pt`
ficam ao lado dos atuais sem sobrescrevê-los:

```
.../datasets/result_decomposed/billboard_voca/
    22050_10.0_5.0/cqt_252_36_2048/    <- ChordMax atual (intacto)
    22050_23.2_11.6/cqt_252_36_512/    <- ChordFormer-like (novo)
```

Comando (no servidor onde estão os áudios + anotações):

```bash
python BTC-ISMIR19/scripts/preprocess_decomposed.py \
    --config BTC-ISMIR19/run_config_chordformer.yaml \
    --datasets billboard dj_avan_songbook1 dj_avan_songbook2 jaah queen robbiewilliams rwc \
    --num_workers 4
```

(Use `scripts/preprocess_datasets.py` se o seu fluxo for o monolítico.)

### Passo 2 — Treinar o backbone ChordFormer-like

```bash
python BTC-ISMIR19/train_decomposed.py \
    --config BTC-ISMIR19/run_config_chordformer.yaml \
    --backbone chordformer \
    --optimizer adamw \
    --scheduler plateau --scheduler_factor 0.1 --scheduler_patience 5 --scheduler_min_lr 1e-6 \
    --crf linear --crf_lambda 30 \
    --disable_gradnorm \
    --batch_size 24 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --num_epochs 200 \
    --kfold 0 \
    --run_name chordformer_replica_k0
```

O treino para automaticamente quando o LR cair abaixo de `1e-6` (igual ao
critério do paper). O `--crf linear --crf_lambda 30` é gravado no
`training_config` do checkpoint para que o passo seguinte saiba qual CRF
instanciar.

### Passo 3 — Treinar (ou só instanciar) o LinearCRF sobre o backbone congelado

```bash
python BTC-ISMIR19/train_harmonic_crf.py \
    --checkpoint checkpoints/chordformer_replica_k0/model_best.pt \
    --config BTC-ISMIR19/run_config_chordformer.yaml \
    --crf_mode root_triad \
    --crf_kind linear --crf_lambda 30 \
    --train_datasets billboard dj_avan_songbook1 dj_avan_songbook2 jaah queen robbiewilliams rwc \
    --kfold 0 \
    --num_epochs 5 \
    --crf_run_name chordformer_replica_k0_linearcrf
```

Como o `LinearCRF` é fixo, não há gradiente para os parâmetros do CRF; o
script ainda roda o loop para gerar métricas comparáveis (loss, accuracy) e
materializar um checkpoint para uso em inferência.

### Passo 4 — Treinar o ChordMax atual (controle)

Rode o ChordMax sem mudanças para a comparação:

```bash
python BTC-ISMIR19/train_decomposed.py \
    --config BTC-ISMIR19/run_config.yaml \
    --backbone chordformer \
    --kfold 0 \
    --run_name chordmax_baseline_k0
```

---

## 3. O que isso não muda

- Os 9 heads de saída do ChordMax (Tabela 4) continuam exatamente os mesmos.
- O pipeline atual (`run_config.yaml` + `train_decomposed.py` sem as flags
  novas) continua produzindo o ChordMax atual, sem regressões.
- Datasets, anotações e pitch-shifting permanecem idênticos.

## 4. O que ficou de fora (declarado)

- Reduzir o output do ChordMax para os 6 heads do ChordFormer
  (Root+Triad conjunto de 84 classes etc.).
- Migrar o split para 60/20/20 do paper.
- Trocar o corpus por Humphrey-Bello.
- Sweep automático de `gamma` ∈ {0.3, 0.5, 0.7, 1.0} e `w_max` ∈ {10, 20} —
  edite `class_weights.gamma`/`w_max` no YAML manualmente.

---

## 5. Arquivos tocados nesta replicação

| Arquivo                                                                   | O que mudou                                                                |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [`run_config_chordformer.yaml`](../run_config_chordformer.yaml)           | YAML novo com os números do ChordFormer.                                   |
| [`utils/transformer_modules.py`](../utils/transformer_modules.py)         | `use_positional_encoding`, `use_batchnorm_in_conv` em ConformerEncoder/Conv. |
| [`models/btc_model_decomposed.py`](../models/btc_model_decomposed.py)     | Repassa os flags do YAML para o encoder.                                   |
| [`utils/hparams.py`](../utils/hparams.py)                                 | `HParams.get()` + `__contains__` / `__getitem__` (destrava `config.get`).  |
| [`models/linear_crf.py`](../models/linear_crf.py)                         | **Novo.** `LinearCRF` com `transitions = lambda * I` (não-treinável).      |
| [`models/harmonic_crf.py`](../models/harmonic_crf.py)                     | `HarmonicCRF` / `FullChordCRF` aceitam `crf_kind` (`trainable` ou `linear`).|
| [`train_decomposed.py`](../train_decomposed.py)                           | Flags `--optimizer`, `--scheduler`, `--crf`, `--disable_gradnorm`, early-stop por LR. |
| [`train_harmonic_crf.py`](../train_harmonic_crf.py)                       | Flags `--crf_kind`, `--crf_lambda` passadas ao `HarmonicCRF` / `FullChordCRF`. |
