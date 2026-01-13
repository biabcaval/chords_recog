# Guia Completo: Treinamento e Inferência do Modelo BTC

Este guia explica como treinar o modelo BTC para reconhecimento de acordes e como realizar inferências após o treinamento.

## 📋 Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Estrutura de Diretórios](#estrutura-de-diretórios)
3. [Treinamento](#treinamento)
4. [Inferência/Teste](#inferênciateste)
5. [Onde os Arquivos são Salvos](#onde-os-arquivos-são-salvos)
6. [Exemplos Práticos](#exemplos-práticos)

---

## Pré-requisitos

### Dependências

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

Principais dependências:
- pytorch >= 1.0.0
- numpy >= 1.16.2
- librosa >= 0.6.3
- mir_eval >= 0.5
- pretty_midi >= 0.2.8

### Estrutura de Dados

Certifique-se de que os datasets estão organizados no diretório configurado em `run_config.yaml`:
- `root_path`: `/home/daniel.melo/datasets`
- Os datasets devem estar pré-processados (use `preprocess_datasets.py` se necessário)

---

## Estrutura de Diretórios

### Diretórios Principais

```
BTC-ISMIR19/
├── assets/                    # Checkpoints e modelos treinados
│   ├── model/                 # Checkpoints padrão (formato antigo)
│   ├── model_1/               # Checkpoints de experimento 1
│   ├── model_2/               # Checkpoints de experimento 2
│   └── tensorboard/           # Logs do TensorBoard
├── RESULTS/                   # Resultados de inferência (.lab e .midi)
├── inferences_*/              # Diretórios customizados de inferência
├── train.py                   # Script de treinamento básico
├── train_curriculum.py        # Script de treinamento com curriculum learning
├── train_kfold.py            # Script para treinar múltiplos k-folds
├── test.py                    # Script de inferência em arquivos de áudio
├── test_full.py              # Script de teste em datasets completos
└── run_config.yaml           # Arquivo de configuração
```

---

## Treinamento

### 1. Configuração Inicial

Edite o arquivo `run_config.yaml` para ajustar os parâmetros:

```yaml
path:
  ckpt_path: 'model'           # Subdiretório para checkpoints
  result_path: '/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/RESULTS'
  asset_path: '/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/assets'
  root_path: '/home/daniel.melo/datasets'

experiment:
  learning_rate: 0.0001
  max_epoch: 100
  batch_size: 128
  save_step: 40                # Salva checkpoint a cada N épocas

feature:
  large_voca: True             # True = 170 acordes, False = 25 acordes (majmin)
```

### 2. Treinamento Básico

**Script:** `train.py`

```bash
python train.py \
    --index 1 \
    --kfold 0 \
    --model btc \
    --dataset1 billboard \
    --dataset2 jaah \
    --test_dataset rwc \
    --voca \
    --early_stop
```

**Parâmetros:**
- `--index`: Número do experimento (usado para nomear checkpoints)
- `--kfold`: Índice do k-fold (0-4)
- `--model`: Tipo de modelo (`btc`, `cnn`, `crnn`)
- `--dataset1`, `--dataset2`: Datasets de treinamento
- `--test_dataset`: Dataset de teste
- `--voca`: Usa vocabulário grande (170 acordes) se True
- `--early_stop`: Para o treinamento se não houver melhoria em 10 épocas

### 3. Treinamento com Curriculum Learning

**Script:** `train_curriculum.py`

```bash
python train_curriculum.py \
    --index 1 \
    --kfold 0 \
    --model btc \
    --dataset1 billboard \
    --dataset2 jaah \
    --test_dataset rwc \
    --voca \
    --curriculum \
    --early_stop
```

O curriculum learning é configurado em `run_config.yaml`:

```yaml
curriculum:
  enabled: True
  strategy: 'mixed'
  pacing: 'linear'
  start_ratio: 0.3
  pace_epochs: 30
```

### 4. Treinamento Múltiplos K-Folds

**Script:** `train_kfold.py`

Treina todos os k-folds (0-4) sequencialmente:

```bash
python train_kfold.py \
    --index 2 \
    --kfold_start 0 \
    --kfold_end 4 \
    --model btc \
    --dataset1 billboard \
    --dataset2 jaah \
    --test_dataset rwc \
    --voca \
    --curriculum \
    --early_stop
```

Este script cria uma estrutura organizada:
```
assets/
└── exp2_btc_billboard_jaah_rwc_voca_curriculum/
    ├── kfold_0/
    │   └── model_037.pth.tar
    ├── kfold_1/
    │   └── model_042.pth.tar
    └── ...
```

### 5. Monitoramento do Treinamento

**TensorBoard:**
```bash
tensorboard --logdir assets/tensorboard/idx_1
```

Visualize:
- Loss de treinamento e validação
- Acurácia de treinamento e validação
- Top-2 accuracy

---

## Inferência/Teste

Após o treinamento, você pode fazer inferências de duas formas:

### 1. Teste em Dataset Completo (com Métricas)

**Script:** `test_full.py`

Este script testa o modelo em um dataset completo e calcula métricas (root, majmin, etc.).

**Edite os parâmetros no início do arquivo:**

```python
CHECKPOINT_PATH = "/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/assets/exp2_btc_billboard_jaah_rwc_voca_curriculum/kfold_2/model_037.pth.tar"
CONFIG_PATH = "run_config.yaml"
TEST_DATASET = "rwc"
KFOLD_INDEX = 2
MODEL_TYPE = "btc"
```

**Execute:**
```bash
python test_full.py
```

**Saída:**
- Métricas de acurácia no console
- Logs detalhados do processo

### 2. Inferência em Arquivos de Áudio

**Script:** `test.py`

Este script processa arquivos de áudio e gera arquivos `.lab` e `.midi` com os acordes reconhecidos.

**Edite os parâmetros no início do arquivo:**

```python
CHECKPOINT_PATH = "/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/assets/exp2_btc_billboard_jaah_rwc_voca_curriculum/kfold_2/model_037.pth.tar"
CONFIG_PATH = "/home/daniel.melo/BTC_ORIGINAL/chords_recog/BTC-ISMIR19/run_config.yaml"
AUDIO_DIR = "/home/daniel.melo/datasets/rwc/audio"
SAVE_DIR = "/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/RESULTS"
KFOLD_INDEX = 2
LARGE_VOCA = True
```

**Execute:**
```bash
python test.py
```

**Saída:**
- Arquivos `.lab`: Anotações de acordes no formato tempo-início tempo-fim acorde
- Arquivos `.midi`: Representação MIDI dos acordes

**Formato do arquivo .lab:**
```
0.000 2.500 C:maj
2.500 5.000 F:maj
5.000 7.500 G:maj
```

---

## Onde os Arquivos são Salvos

### Checkpoints (Modelos Treinados)

**Formato Antigo (train.py):**
```
assets/
└── model/
    ├── idx_1_001.pth.tar
    ├── idx_1_002.pth.tar
    └── ...
```

**Formato Novo (train_curriculum.py / train_kfold.py):**
```
assets/
└── exp{index}_{model}_{datasets}_{voca}_{curriculum}/
    └── kfold_{num}/
        ├── model_001.pth.tar
        ├── model_002.pth.tar
        └── ...
```

**Estrutura do Checkpoint:**
```python
{
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch_number
}
```

### Arquivos de Normalização

Os arquivos de normalização (mean e std) são salvos em:
```
{root_path}/result/{mp3_string}_{feature_string}mix_kfold_{kfold}_normalization.pt
```

Exemplo:
```
/home/daniel.melo/datasets/result/22050_10.0_5.0_cqt_144_24_2048_mix_kfold_2_normalization.pt
```

**Importante:** Este arquivo é necessário para fazer inferências! Ele contém a média e desvio padrão usados na normalização durante o treinamento.

### Resultados de Inferência

**Diretório padrão (test.py):**
```
/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/RESULTS/
├── song1.lab
├── song1.midi
├── song2.lab
└── song2.midi
```

**Diretórios customizados:**
Você pode criar diretórios customizados para organizar inferências:
```
/home/daniel.melo/BTC_ORIGINAL/chords_recog/BTC-ISMIR19/inferences_1trainBillJaah_testRwc/
├── rwc-pop_001.lab
├── rwc-pop_001.midi
└── ...
```

### Logs do TensorBoard

```
assets/tensorboard/
├── idx_1/
│   └── events.out.tfevents.*
└── idx_2/
    └── events.out.tfevents.*
```

---

## Exemplos Práticos

### Exemplo 1: Treinamento Completo

```bash
# 1. Treinar modelo com k-fold 0
python train_curriculum.py \
    --index 1 \
    --kfold 0 \
    --model btc \
    --dataset1 billboard \
    --dataset2 jaah \
    --test_dataset rwc \
    --voca \
    --curriculum \
    --early_stop

# 2. Após o treinamento, o melhor checkpoint será salvo em:
# assets/exp1_btc_billboard_jaah_rwc_voca_curriculum/kfold_0/model_XXX.pth.tar

# 3. Fazer inferência no dataset RWC
# Edite test_full.py com o caminho do checkpoint e execute:
python test_full.py
```

### Exemplo 2: Inferência em Áudio Customizado

```bash
# 1. Coloque seus arquivos de áudio em um diretório
mkdir -p /home/daniel.melo/my_audio_files
# Copie arquivos .mp3 ou .wav para este diretório

# 2. Edite test.py:
# - CHECKPOINT_PATH: caminho do checkpoint treinado
# - AUDIO_DIR: "/home/daniel.melo/my_audio_files"
# - SAVE_DIR: diretório onde salvar resultados
# - KFOLD_INDEX: mesmo k-fold usado no treinamento

# 3. Execute
python test.py

# 4. Resultados estarão em SAVE_DIR
```

### Exemplo 3: Treinar Todos os K-Folds

```bash
# Treina k-folds 0, 1, 2, 3, 4 sequencialmente
python train_kfold.py \
    --index 2 \
    --kfold_start 0 \
    --kfold_end 4 \
    --model btc \
    --dataset1 billboard \
    --dataset2 jaah \
    --test_dataset rwc \
    --voca \
    --curriculum

# Depois teste cada k-fold:
# Edite test_full.py para cada checkpoint e execute
```

### Exemplo 4: Usar Checkpoint Específico

```python
# Em test_full.py ou test.py, defina:
CHECKPOINT_PATH = "/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/assets/exp2_btc_billboard_jaah_rwc_voca_curriculum/kfold_2/model_037.pth.tar"

# Certifique-se de usar o mesmo KFOLD_INDEX usado no treinamento:
KFOLD_INDEX = 2
```

---

## Dicas Importantes

1. **K-Fold Index:** Sempre use o mesmo `kfold` index no treinamento e na inferência, pois o arquivo de normalização é específico para cada k-fold.

2. **Large Voca:** Se treinou com `--voca`, use `LARGE_VOCA = True` na inferência. Se treinou sem, use `LARGE_VOCA = False`.

3. **Arquivo de Normalização:** Este arquivo é criado automaticamente durante o treinamento. Certifique-se de que ele existe antes de fazer inferências.

4. **Checkpoints:** O script de treinamento salva:
   - Checkpoint do melhor modelo (melhor acurácia de validação)
   - Checkpoints periódicos (a cada `save_step` épocas)

5. **Estrutura de Pastas de Inferência:** Você pode criar diretórios customizados para organizar diferentes experimentos de inferência, como:
   - `inferences_1trainBillJaah_testRwc/`
   - `inferences_2trainBillJaahDjavan_testRwc/`

---

## Troubleshooting

### Erro: "Normalization file not found"
- Certifique-se de que o arquivo de normalização existe no caminho esperado
- Verifique se o `KFOLD_INDEX` está correto
- O arquivo é criado durante o primeiro treinamento

### Erro: "Checkpoint not found"
- Verifique o caminho do checkpoint
- Certifique-se de que o treinamento foi concluído
- Verifique se está usando o caminho correto (formato antigo vs novo)

### Erro: "Dataset path does not exist"
- Verifique se os datasets estão pré-processados
- Execute `preprocess_datasets.py` se necessário
- Verifique o `root_path` em `run_config.yaml`

---

## Resumo Rápido

**Treinar:**
```bash
python train_curriculum.py --index 1 --kfold 0 --model btc --dataset1 billboard --dataset2 jaah --test_dataset rwc --voca --curriculum
```

**Testar (métricas):**
```bash
# Edite test_full.py com CHECKPOINT_PATH e execute:
python test_full.py
```

**Inferir (áudio):**
```bash
# Edite test.py com CHECKPOINT_PATH, AUDIO_DIR, SAVE_DIR e execute:
python test.py
```

**Checkpoints salvos em:**
- `assets/exp{index}_.../kfold_{num}/model_XXX.pth.tar`

**Resultados salvos em:**
- `RESULTS/` (padrão) ou diretório customizado definido em `SAVE_DIR`

---

**Última atualização:** Dezembro 2024

