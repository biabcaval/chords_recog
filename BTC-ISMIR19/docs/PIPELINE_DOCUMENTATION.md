# Pipeline de Reconhecimento de Acordes com Decomposição Estrutural

Este documento descreve todo o processo desde o pré-processamento do áudio até a saída final do modelo, utilizando a técnica de **Chord Structure Decomposition (CSD)**.

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Pré-processamento](#2-pré-processamento)
3. [Decomposição de Acordes](#3-decomposição-de-acordes)
4. [Carregamento de Dados](#4-carregamento-de-dados)
5. [Arquitetura do Modelo](#5-arquitetura-do-modelo)
6. [Função de Perda](#6-função-de-perda)
7. [Treinamento](#7-treinamento)
8. [Inferência e Reassemblagem](#8-inferência-e-reassemblagem)
9. [Métricas de Avaliação](#9-métricas-de-avaliação)
10. [Debug e Testes](#10-debug-e-testes)

---

## 1. Visão Geral

### Objetivo
Reconhecer acordes musicais a partir de áudio, decompondo cada acorde em **9 componentes** estruturais independentes, ao invés de classificar diretamente em uma das 170+ classes de acordes.

### Motivação
- **Problema da cauda longa**: Acordes complexos (como `C:maj9(#11)`) são raros no dataset
- **Compartilhamento de conhecimento**: O modelo aprende que `C:maj7` e `D:maj7` compartilham a mesma estrutura de tríade e sétima
- **Melhor generalização**: Componentes individuais têm distribuições mais balanceadas

### Fluxo Geral

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Áudio     │───▶│    CQT       │───▶│ ChordMax    │───▶│  9 Saídas    │
│   (.wav)    │    │  Features    │    │ (Conformer) │    │  (softmax)   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                 │
                                                                 ▼
                                                         ┌──────────────┐
                                                         │ Reassembler  │
                                                         │  C:maj7      │
                                                         └──────────────┘
```

---

## 2. Pré-processamento

### 2.1 Extração de Features (CQT)

O áudio é convertido em uma representação tempo-frequência usando **Constant-Q Transform (CQT)**.

**Parâmetros (de `run_config.yaml`):**

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `song_hz` | 22050 | Taxa de amostragem |
| `n_bins` | 252 | Número de bins de frequência |
| `bins_per_octave` | 36 | Bins por oitava (3 por semitom) |
| `hop_length` | 2048 | Salto entre frames (~93ms) |

**Cálculo:**
```python
# Tempo por frame
frame_duration = hop_length / song_hz  # 2048/22050 ≈ 0.093s

# Frequência mínima (cobrindo 6 oitavas a partir de C1)
fmin = librosa.note_to_hz('C1')  # ~32.7 Hz
```

**Saída:** Matriz `(n_bins, n_frames)` = `(252, T)`

### 2.2 Segmentação

O áudio é dividido em segmentos de tamanho fixo para o treinamento.

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `timestep` | 108 | Número de frames por segmento |
| `duration` | ~10s | Duração do segmento |

**Fórmula:**
```
duração = timestep × (hop_length / song_hz)
duração = 108 × (2048 / 22050) ≈ 10.03 segundos
```

### 2.3 Normalização

As features passam por dois estágios de normalização:

**1. Log-magnitude (sempre aplicado):**

```python
features = np.log(np.abs(cqt) + 1e-6)
```

**2. Padronização mean/std (opcional, recomendado):**

```python
features = (features - mean) / std
```

Os valores de mean e std são computados **uma única vez** sobre todos os dados dos datasets de **treino** (sem incluir dados de teste, para evitar data leakage):

```bash
python scripts/compute_normalization.py \
    --config run_config.yaml \
    --datasets billboard dj_avan jaah \
    --output normalization_BiDjJa.pt
```

O script varre 100% dos arquivos `.pt` dos datasets indicados (sem split de k-fold) e salva um arquivo com dois escalares:

```python
{
    'mean': -3.245,      # média global das features log-CQT
    'std': 1.872,        # desvio padrão global
    'datasets': ['billboard', 'dj_avan', 'jaah'],
    'n_files': 523048,
}
```

**Regras importantes:**

- Cada combinação de datasets de treino precisa do seu próprio `normalization.pt`
- Nunca incluir o dataset de teste no cálculo (data leakage)
- Os valores são salvos automaticamente dentro dos checkpoints pelo `train_decomposed.py`
- Na inferência, mean/std são carregados do checkpoint — sem necessidade de arquivo extra

**Uso no treino:**

```bash
python train_decomposed.py \
    --normalization normalization_BiDjJa.pt \
    --train_datasets billboard dj_avan jaah \
    --backbone chordformer \
    --kfold 0
```

A normalização é aplicada on-the-fly no `AudioDatasetStructured.__getitem__()` a cada sample carregado. Se `--normalization` não for passado, o treino usa features sem padronização (apenas log-magnitude), mantendo retrocompatibilidade.

### 2.4 Estrutura dos Arquivos `.pt`

Cada segmento é salvo como um arquivo PyTorch:

```python
{
    'feature': np.array,              # Shape: (252, 108) - CQT features
    'chord': list,                    # Lista de índices de acordes por frame
    'original_chords': list,          # Índices originais (backup)
    'original_chord_labels': list,    # Labels originais com extensões (ex: 'C:maj7(9)')
}
```

> **Nota**: O campo `original_chord_labels` contém os labels originais dos arquivos `.lab` 
> com extensões completas (ex: `C:maj7(9)`, `B:7(b9)`). Este campo é adicionado pelo script
> `scripts/add_original_labels.py` e permite capturar extensões 9th, 11th, 13th que são
> simplificadas no vocabulário padrão de 170 classes.

**Localização:** `/datasets/result/{dataset}_voca/22050_10.0_5.0/cqt_252_36_2048/{song}/`

### 2.5 Adicionando Labels Originais aos Arquivos .pt

O pré-processamento original simplifica as anotações de acordes, perdendo extensões como `(9)`, `(b9)`, `(#11)`. Para recuperar essas extensões:

```bash
# Dry-run (apenas mostra o que seria feito)
python scripts/add_original_labels.py --data_root /path/to/datasets --dry_run

# Executar (adiciona o campo original_chord_labels aos .pt)
python scripts/add_original_labels.py --data_root /path/to/datasets
```

**O que o script faz:**
1. Encontra todos os arquivos `.pt` no diretório de dados
2. Para cada arquivo, localiza o arquivo `.lab` correspondente
3. Para cada frame do segmento, extrai o label do acorde original baseado no timestamp
4. Adiciona o campo `original_chord_labels` ao arquivo `.pt`

**Exemplo de mapeamento:**
```
Arquivo .lab:                    Arquivo .pt (após script):
0.000  2.901 C:maj7(9)    →     original_chord_labels: ['C:maj7(9)', 'C:maj7(9)', ...]
2.902  4.271 F#:hdim7     →     (um label por frame de 93ms)
4.272  5.920 B:7(b9)
```

---

## 3. Decomposição de Acordes

### 3.1 Os 9 Componentes

Cada acorde é decomposto em **9 componentes** independentes:

| # | Componente | Classes | Vocabulário |
|---|------------|---------|-------------|
| 1 | **Root** | 13 | N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B |
| 2 | **Bass** | 13 | N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B |
| 3 | **Triad** | 7 | N, maj, min, dim, aug, sus2, sus4 |
| 4 | **Misc** | 2 | N, 5 (power chord) |
| 5 | **6th** | 2 | N, 6 |
| 6 | **7th** | 4 | N, 7, b7, bb7 |
| 7 | **9th** | 4 | N, 9, #9, b9 |
| 8 | **11th** | 3 | N, 11, #11 |
| 9 | **13th** | 3 | N, 13, b13 |

**Total de classes:** 13 + 13 + 7 + 2 + 2 + 4 + 4 + 3 + 3 = **51**

**Total de combinações possíveis:** 13 × 13 × 7 × 2 × 2 × 4 × 4 × 3 × 3 = **307,008**

### 3.2 Exemplos de Decomposição

| Acorde | Root | Bass | Triad | Misc | 6th | 7th | 9th | 11th | 13th |
|--------|------|------|-------|------|-----|-----|-----|------|------|
| `C:maj` | C | N | maj | N | N | N | N | N | N |
| `G:min7` | G | N | min | N | N | b7 | N | N | N |
| `D:7` | D | N | maj | N | N | b7 | N | N | N |
| `A:min7/E` | A | E | min | N | N | b7 | N | N | N |
| `F:maj7` | F | N | maj | N | N | 7 | N | N | N |
| `C:maj7(9)` | C | N | maj | N | N | 7 | 9 | N | N |
| `B:7(b9)` | B | N | maj | N | N | b7 | b9 | N | N |
| `A:min6` | A | N | min | N | 6 | N | N | N | N |
| `C:13` | C | N | maj | N | N | b7 | 9 | 11 | 13 |
| `E:5` | E | N | N | 5 | N | N | N | N | N |
| `N` | N | N | N | N | N | N | N | N | N |

### 3.3 Definições de Intervalos

```python
INTERVAL_DEFINITIONS = {
    'maj':  [0, 4, 7],      # 1, 3, 5
    'min':  [0, 3, 7],      # 1, b3, 5
    'dim':  [0, 3, 6],      # 1, b3, b5
    'aug':  [0, 4, 8],      # 1, 3, #5
    'sus2': [0, 2, 7],      # 1, 2, 5
    'sus4': [0, 5, 7],      # 1, 4, 5
    '5':    [0, 7],         # 1, 5 (power chord)
}
```

### 3.4 Código de Decomposição

```python
# utils/chord_decomposition.py

class ChordDecomposer:
    def decompose(self, chord_label: str) -> Dict[str, str]:
        """
        Decompõe 'C:maj7' em:
        {
            'root': 'C',
            'bass': 'N',
            'triad': 'maj',
            'misc': 'N',
            '6th': 'N',
            '7th': '7',
            '9th': 'N',
            '11th': 'N',
            '13th': 'N'
        }
        
        Também suporta extensões parentéticas dos arquivos .lab:
        'C:maj7(9)' -> 7th='7', 9th='9'
        'B:7(b9)'   -> 7th='b7', 9th='b9'
        """
```

---

## 4. Carregamento de Dados

### 4.1 AudioDatasetStructured

Estende o `AudioDataset` base para incluir decomposição:

```python
# data/audio_dataset_structured.py

class AudioDatasetStructured(Dataset):
    def __getitem__(self, idx):
        # 1. Carrega arquivo .pt
        data = torch.load(instance_path)
        
        # 2. Processa features
        features = np.log(np.abs(data['feature']) + 1e-6)
        features = features.T  # (T, 252)
        
        # 3. Converte índices para labels
        chord_labels = self._get_chord_labels(data['chord'])
        # ['130', '130', ...] → ['A:min6', 'A:min6', ...]
        
        # 4. Decompõe em 9 componentes
        components = self.decomposer.decompose_batch(chord_labels)
        # {'root': [9, 9, ...], 'triad': [2, 2, ...], ...}
        
        return {
            'feature': torch.FloatTensor(features),  # (108, 252)
            'chord': chord_labels,                    # lista original
            'components': components                  # dict de tensores
        }
```

### 4.2 Collate Function

Agrupa amostras em batches:

```python
def _collate_fn_structured(batch):
    # Stack features: (B, T, F) = (batch, 108, 252)
    features = torch.stack([s['feature'] for s in batch])
    
    # Stack components: (B, T) para cada componente
    components = {
        name: torch.stack([s['components'][name] for s in batch])
        for name in COMPONENT_NAMES
    }
    
    return {
        'features': features,      # (B, 108, 252)
        'components': components   # Dict[str, Tensor(B, 108)]
    }
```

### 4.3 Fluxo de Dados

```
┌──────────────────────────────────────────────────────────────────┐
│                        DataLoader                                 │
├──────────────────────────────────────────────────────────────────┤
│  Arquivo .pt                                                      │
│  ├── feature: (252, T)                                           │
│  ├── chord: ['130', '130', '45', ...]  (índices como strings)    │
│  └── original_chords: [...]                                      │
│                           │                                       │
│                           ▼                                       │
│  idx2chord mapping:  130 → 'A:min6'                              │
│                           │                                       │
│                           ▼                                       │
│  ChordDecomposer:  'A:min6' → {root:'A', triad:'min', 7th:'N'...}│
│                           │                                       │
│                           ▼                                       │
│  to_indices:  {root: 10, bass: 0, triad: 2, ...}                 │
│                           │                                       │
│                           ▼                                       │
│  Batch Output:                                                    │
│  ├── features: (B, 108, 252)                                     │
│  └── components:                                                  │
│      ├── root:  (B, 108) - valores 0-12                          │
│      ├── bass:  (B, 108) - valores 0-12                          │
│      ├── triad: (B, 108) - valores 0-6                           │
│      └── ...                                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Arquitetura do Modelo

### 5.1 Visão Geral

```
┌─────────────────────────────────────────────────────────────────┐
│              ChordMax (ChordFormer_model_decomposed)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: (B, T, F) = (batch, 108, 252)                           │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ConformerEncoder (12 blocos)               │    │
│  │  - Input projection: Linear(252 → 128)                 │    │
│  │  - Positional encoding                                  │    │
│  │  - 12× ConformerBlock (Macaron-style):                  │    │
│  │    x + 0.5*FFN → x + MHSA → x + Conv → x + 0.5*FFN    │    │
│  │  - Output: (B, T, hidden_size=128)                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Multi-Head Output (9 cabeças)               │    │
│  │                                                          │    │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌───┐ ┌───┐ ┌────┐│    │
│  │  │ Root │ │ Bass │ │Triad │ │ Misc │ │7th│ │9th│ │... ││    │
│  │  │  13  │ │  13  │ │  7   │ │  2   │ │ 4 │ │ 4 │ │    ││    │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └───┘ └───┘ └────┘│    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  Output: Dict[component_name → Tensor(B, T, vocab_size)]        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Componentes do Modelo

#### 5.2.1 ConformerEncoder

O ChordMax usa um encoder Conformer com 12 blocos Macaron-style. Cada bloco tem 4 conexões residuais:

```python
class ConformerBlock(nn.Module):
    def forward(self, x):
        x = x + 0.5 * self.ff1(x)        # Half-step FFN
        x = x + self.self_attn(x)         # Multi-head self-attention
        x = x + self.conv_module(x)       # Depthwise convolution
        x = x + 0.5 * self.ff2(x)        # Half-step FFN
        x = self.layer_norm(x)
        return x

class ConformerEncoder(nn.Module):
    def __init__(self, ...):
        self.input_projection = nn.Linear(252, 128)  # feature_size → hidden_size
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(...) for _ in range(12)
        ])
```

**Parâmetros do Conformer:**
- `conv_kernel_size=31`: contexto temporal amplo na conv depthwise
- `ff_expansion_factor=4`: FFN interna de 128 → 512 → 128
- `conv_expansion_factor=2`: expansão no módulo convolucional
- 12 blocos × 4 residuais = 48 conexões residuais no total

#### 5.2.2 Component Head

Cada componente tem sua própria cabeça de classificação, com dois modos de operação:

**Modo simples (padrão):** projeção direta para o vocabulário.

```python
# use_head_ffn = False (default)
Dropout -> Linear(hidden_size, vocab_size)
```

**Modo FFN bottleneck (opcional):** adiciona uma FFN com compressão progressiva antes da projeção final. Ativado via `--use_head_ffn` no treino.

```python
# use_head_ffn = True, head_ffn_dim = hidden_size // 2 (default = 64)
Linear(128, 64) -> ReLU -> Dropout -> Linear(64, 32) -> ReLU -> Linear(32, vocab_size)
```

```python
class ComponentHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout=0.0,
                 use_ffn=False, ffn_dim=None):
        # Modo FFN: bottleneck com compressão progressiva
        if use_ffn:
            ffn_dim = ffn_dim or hidden_size // 2
            bottleneck_dim = ffn_dim // 2
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, ffn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, vocab_size),
            )
        # Modo simples: projeção direta
        else:
            self.linear = nn.Linear(hidden_size, vocab_size)
            self.dropout = nn.Dropout(dropout)
```

A escolha entre os dois modos é controlada pelo parâmetro `use_head_ffn` no config/CLI e é salva automaticamente no checkpoint para que a inferência reconstrua a arquitetura correta.

#### 5.2.3 Multi-Head Decomposer

```python
class MultiHeadChordDecomposer(nn.Module):
    def __init__(self, hidden_size):
        self.heads = nn.ModuleDict({
            'root':  ComponentHead(hidden_size, 13),
            'bass':  ComponentHead(hidden_size, 13),
            'triad': ComponentHead(hidden_size, 7),
            'misc':  ComponentHead(hidden_size, 2),
            '7th':   ComponentHead(hidden_size, 4),
            '9th':   ComponentHead(hidden_size, 4),
            '11th':  ComponentHead(hidden_size, 3),
            '13th':  ComponentHead(hidden_size, 3),
        })
    
    def forward(self, x):
        return {name: head(x) for name, head in self.heads.items()}
```

### 5.3 Parâmetros do Modelo

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `feature_size` | 252 | Dimensão das features CQT (= `n_bins`) |
| `hidden_size` | 128 | Dimensão interna |
| `num_layers` | 12 | Blocos do encoder |
| `num_heads` | 4 | Cabeças de atenção |
| `input_dropout` | 0.2 | Dropout na entrada do encoder |
| `layer_dropout` | 0.2 | Dropout dentro dos blocos |
| `output_dropout` | 0.0 | Dropout nas cabeças de saída |
| `conv_kernel_size` | 31 | Kernel da conv depthwise (ChordFormer) |
| `ff_expansion_factor` | 4 | Expansão da FFN (ChordFormer) |
| `use_head_ffn` | False | Ativar FFN bottleneck nas output heads (ChordFormer only) |
| `head_ffn_dim` | hidden_size//2 | Dimensão oculta da FFN nas heads (default: 64 quando hidden=128) |

**Total de parâmetros:** ~4.65M sem FFN heads / ~4.68M com FFN heads (ChordFormer, 12 layers, hidden=128)

---

## 6. Função de Perda

### 6.1 Multi-Task Loss

A perda total é a soma ponderada das perdas de cada componente:

```python
class MultiTaskLoss(nn.Module):
    def forward(self, outputs, targets):
        total_loss = 0
        
        for component in COMPONENT_NAMES:
            # outputs[component]: (B, T, num_classes)
            # targets[component]: (B, T)
            
            logits = outputs[component].view(-1, num_classes)
            labels = targets[component].view(-1)
            
            loss = F.cross_entropy(
                logits, 
                labels, 
                weight=self.class_weights[component]
            )
            
            total_loss += loss
        
        return total_loss
```

### 6.2 Class Re-weighting

Para lidar com classes desbalanceadas (acordes raros):

**Fórmula:**
$$w_m^{(j)} = \min\left(\left(\frac{n_m^{(j)}}{\max_{m'} n_{m'}^{(j)}}\right)^{-\gamma}, w_{max}\right)$$

Onde:
- $n_m^{(j)}$ = contagem da classe $m$ no componente $j$
- $\gamma = 0.5$ = expoente de suavização
- $w_{max} = 10$ = peso máximo

**Exemplo:**
```
Componente '7th':
  - 'N':   80,000 frames → weight = 1.0
  - '7':   15,000 frames → weight = 2.3
  - 'b7':   4,500 frames → weight = 4.2
  - 'bb7':    500 frames → weight = 10.0 (capped)
```

### 6.3 GradNorm (Balanceamento Adaptativo de Tarefas)

O GradNorm ajusta dinamicamente os pesos `w_i` de cada componente durante o treino, baseado na velocidade relativa de aprendizado de cada tarefa.

**Algoritmo (por batch):**
1. Calcula perda bruta `L_i` por componente.
2. Mede norma de gradiente `G_i = ||∇_W (w_i L_i)||` na camada compartilhada final.
3. Calcula taxa relativa `r_i = (L_i/L_i(0)) / mean(L_j/L_j(0))`.
4. Define alvo `G_i* = mean(G) * r_i^alpha`.
5. Minimiza `L_grad = Σ|G_i - G_i*|` atualizando apenas `w_i`.
6. Renormaliza `Σw_i = 9` (T = número de tarefas).

**Hiperparâmetros:**

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `alpha` | 1.5 | Força de balanceamento (maior = mais agressivo) |
| `lr` | 0.025 | Learning rate dos pesos GradNorm |
| `eps` | 1e-8 | Estabilidade numérica |
| `w_min` | 1e-3 | Peso mínimo por tarefa |
| `w_max` | 10.0 | Peso máximo por tarefa |

**Uso via CLI:**

```bash
python train_decomposed.py \
    --use_gradnorm \
    --gradnorm_alpha 1.5 \
    --gradnorm_lr 0.025
```

**Configuração em `run_config.yaml`:**

```yaml
gradnorm:
  enabled: False
  alpha: 1.5
  lr: 0.025
  eps: 1.0e-8
  w_min: 1.0e-3
  w_max: 10.0
```

### 6.6.1 Focal Loss

A Focal Loss (Lin et al., 2017) adiciona modulação por dificuldade de amostra sobre o CrossEntropyLoss. O fator `(1 - pt)^γ` reduz a contribuição de amostras que o modelo já classifica com alta confiança, focando o treinamento em amostras difíceis.

**Fórmula:**

```
FL(pt) = -αt · (1 - pt)^γ · log(pt)
```

onde `pt` é a probabilidade softmax atribuída à classe correta, `γ` controla a intensidade da modulação e `αt` são os class weights (mesmos já usados pelo sistema de class re-weighting).

**Complementaridade com os outros mecanismos:**

| Nível | Mecanismo | O que balanceia |
|-------|-----------|-----------------|
| Classe | Class weights (`αt`) | Frequência (classes raras vs comuns) |
| Amostra | Focal Loss `(1-pt)^γ` | Dificuldade (amostras fáceis vs difíceis) |
| Tarefa | GradNorm | Ritmo de aprendizado entre os 9 componentes |

Quando `focal_gamma=0.0`, o comportamento é idêntico ao CrossEntropyLoss padrão (backward compatible).

**Cuidado de implementação:** quando focal está ativo, os class weights são aplicados como `alpha` (gathered per-sample) e **não** passados ao `weight` do `F.cross_entropy`, pois isso distorceria o cálculo de `pt`.

**Uso via CLI:**

```bash
python train_decomposed.py \
    --focal_gamma 2.0
```

**Configuração em `run_config.yaml`:**

```yaml
focal:
  gamma: 0.0
```

---

## 7. Treinamento

### 7.1 Comandos de Treinamento


#### Treino Básico

```bash
# Treino ChordMax com nome automático da run
python train_decomposed.py --backbone chordformer
```

#### Treino com Nome Personalizado

```bash
python train_decomposed.py --backbone chordformer --run_name meu_experimento
```

#### Treino Completo (recomendado)

```bash
python train_decomposed.py \
    --config run_config.yaml \
    --backbone chordformer \
    --kfold 0 \
    --run_name cf_gradnorm_k0 \
    --num_epochs 100 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --no_class_weights \
    --use_gradnorm \
    --gradnorm_alpha 1.5 \
    --gradnorm_lr 0.025 \
    --wandb_project chordMax \
    --wandb_entity teste-time
```

#### Treino com datasets específicos (override do config)

```bash
# Treina/valida apenas com billboard+jaah+queen (ignora config)
python train_decomposed.py \
    --config run_config.yaml \
    --backbone chordformer \
    --kfold 0 \
    --train_datasets billboard jaah queen \
    --run_name cf_3ds_k0 \
    --num_epochs 100 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --use_gradnorm
```

Os datasets passados via `--train_datasets` são usados tanto para treino (4 folds) quanto para validação (1 fold), seguindo a lógica de k-fold.

#### Treino com FFN Bottleneck nas Output Heads

```bash
# FFN com dimensão padrão (hidden_size // 2 = 64)
python train_decomposed.py \
    --backbone chordformer \
    --use_head_ffn \
    --run_name cf_headffn_k0

# FFN com dimensão customizada
python train_decomposed.py \
    --backbone chordformer \
    --use_head_ffn \
    --head_ffn_dim 96 \
    --run_name cf_headffn96_k0
```

A FFN bottleneck adiciona uma rede `Linear(128→ffn_dim) → ReLU → Dropout → Linear(ffn_dim→ffn_dim//2) → ReLU → Linear(ffn_dim//2→vocab)` em cada uma das 9 output heads, substituindo a projeção direta `Linear(128→vocab)`. A configuração é salva no checkpoint e restaurada automaticamente na inferência.

#### Treino Rápido (smoke test)

```bash
python quick_test_decomposed.py --backbone chordformer
```

### 7.2 Parâmetros do Script

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `--run_name` | auto (timestamp) | Nome da run (cria subpasta) |
| `--config` | run_config.yaml | Arquivo de configuração |
| `--device` | cuda | Dispositivo (cuda, cpu) |
| `--output_dir` | ./checkpoints | Diretório base para saves |
| `--num_epochs` | 100 | Número de épocas |
| `--batch_size` | 32 | Tamanho do batch |
| `--learning_rate` | 0.001 | Taxa de aprendizado |
| `--weight_decay` | 1e-5 | Regularização L2 |
| `--gamma` | 0.5 | Expoente do class weighting |
| `--w_max` | 10.0 | Peso máximo por classe |
| `--backbone` | chordformer | Backbone do modelo (`chordformer`) |
| `--use_head_ffn` | - | Ativar FFN bottleneck nas output heads (ChordFormer only) |
| `--head_ffn_dim` | hidden_size//2 | Dimensão oculta da FFN nas heads |
| `--kfold` | 4 | Índice do k-fold para validação (0-4) |
| `--log_interval` | 10 | Intervalo de log (batches) |
| `--val_interval` | 1 | Intervalo de validação (epochs) |
| `--resume` | None | Checkpoint para continuar treino |
| `--use_gradnorm` | - | Ativar GradNorm |
| `--no_gradnorm` | - | Desativar GradNorm |
| `--gradnorm_alpha` | 1.5 | Força de balanceamento GradNorm |
| `--gradnorm_lr` | 0.025 | Learning rate dos pesos GradNorm |
| `--gradnorm_w_min` | 1e-3 | Peso mínimo por tarefa GradNorm |
| `--gradnorm_w_max` | 10.0 | Peso máximo por tarefa GradNorm |
| `--focal_gamma` | 0.0 | Focal loss gamma (0=CE padrão, 2=recomendado) |
| `--component_weights` | None | Pesos estáticos por componente (`root=1,11th=0.3,...`) |
| `--use_class_weights` | - | Forçar rebalanceamento por classe |
| `--no_class_weights` | - | Desativar rebalanceamento por classe |
| `--class_weights_mode` | auto | Estratégia de class weights (`auto`, `compute`, `load`) |
| `--train_datasets` | config | Datasets para treino/validação (ex: `billboard jaah queen`). Override do config |
| `--wandb_project` | chordMax | Projeto no Weights & Biases |
| `--wandb_entity` | None | Entidade/time no W&B |
| `--wandb_disabled` | - | Desativar logging W&B |

### 7.3 Estrutura de Checkpoints

Os checkpoints são organizados por run:

```
checkpoints/
├── meu_experimento/
│   ├── model_best.pt           # Melhor modelo
│   ├── model_best_info.json    # Metadados legíveis
│   ├── model_epoch_010.pt      # Checkpoint época 10
│   ├── model_epoch_020.pt      # Checkpoint época 20
│   ├── model_final.pt          # Modelo final
│   └── training_history.json   # Histórico de loss
├── baseline_v1/
│   └── ...
└── run_20260205_143052/        # Nome automático
    └── ...
```

### 7.4 Conteúdo do Checkpoint

Cada checkpoint `.pt` contém:

```python
{
    'epoch': 15,                    # Época atual
    'total_epochs': 100,            # Total de épocas
    'model_state_dict': {...},      # Pesos do modelo
    'optimizer_state_dict': {...},  # Estado do otimizador
    'scheduler_state_dict': {...},  # Estado do scheduler
    
    'metrics': {
        'train_loss': 1.2345,
        'val_loss': 1.4567,
        'component_losses': {
            'root': 0.15,
            'triad': 0.18,
            ...
        }
    },
    
    'training_config': {
        'run_name': 'meu_experimento',
        'learning_rate': 0.001,
        'batch_size': 32,
        'model_config': {
            'hidden_size': 128,
            'num_layers': 12,
            ...
        },
        'datasets': ['billboard', 'dj_avan', ...],
        'train_samples': 17736,
        'val_samples': 5076,
    },
    
    'saved_at': '2026-02-05T14:30:23'
}
```

### 7.5 Monitoramento do Treino

O treino mostra progresso em tempo real:

```
Run name: meu_experimento
Output directory: checkpoints/meu_experimento

=== Epoch 1/100 ===
Batch 58/581, Loss: 3.2145
Batch 116/581, Loss: 2.8934
...
Train Loss: 2.5432
Val Loss: 2.7891
Saved best checkpoint to checkpoints/meu_experimento/model_best.pt
Saved checkpoint info to checkpoints/meu_experimento/model_best_info.json
```

### 7.6 Carregar Checkpoint

```python
import torch

# Carregar checkpoint
checkpoint = torch.load('checkpoints/meu_experimento/model_best.pt')

# Ver informações
print(f"Run: {checkpoint['training_config']['run_name']}")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Loss: {checkpoint['metrics']['val_loss']:.4f}")
print(f"Learning Rate: {checkpoint['training_config']['learning_rate']}")
print(f"Hidden Size: {checkpoint['training_config']['model_config']['hidden_size']}")

# Carregar modelo
model.load_state_dict(checkpoint['model_state_dict'])
```

### 7.7 Configuração (run_config.yaml)

```yaml
# run_config.yaml
experiment:
  data_root: /home/daniel.melo/datasets
  dataset_names: ['billboard', 'dj_avan', 'jaah', 'queen', 'robbiewilliams', 'rwc']
  learning_rate: 0.0001
  max_epoch: 100
  batch_size: 128

model:
  feature_size: 252
  hidden_size: 128
  num_layers: 12
  num_heads: 4
  input_dropout: 0.2
  layer_dropout: 0.2
  output_dropout: 0.0
  conv_kernel_size: 31
  ff_expansion_factor: 4
  conv_expansion_factor: 2
  use_head_ffn: False           # FFN bottleneck nas output heads (ChordFormer only)
  head_ffn_dim: null            # Dim da FFN (default: hidden_size // 2)

class_weights:
  enabled: True
  gamma: 0.5
  w_max: 10.0

gradnorm:
  enabled: False
  alpha: 1.5
  lr: 0.025
  eps: 1.0e-8
  w_min: 1.0e-3
  w_max: 10.0

focal:
  gamma: 0.0
```

### 7.7.1 Cache de Class Weights (novo fluxo recomendado)

Para reduzir o tempo de startup do treino, desacoplamos o cálculo de class weights do loop principal.

Pré-cálculo offline:

```bash
python scripts/precompute_class_weights_decomposed.py \
    --config run_config.yaml \
    --kfold 0 \
    --gamma 0.5 \
    --w_max 10.0
```

Treino usando cache pré-computado:

```bash
python train_decomposed.py \
    --config run_config.yaml \
    --kfold 0 \
    --use_class_weights \
    --class_weights_mode load
```

Novos argumentos em `train_decomposed.py`:

- `--class_weights_mode auto|compute|load` (default: `auto`)
- `--class_weights_path` (arquivo `.pt` explícito)
- `--class_weights_cache_dir` (default: `./class_weights_cache`)

### 7.8 Loop de Treinamento (Interno)

```python
for epoch in range(num_epochs):
    model.train()
    
    for batch in train_loader:
        features = batch['features'].to(device)      # (B, T, F)
        components = batch['components']              # Dict[str, (B, T)]
        
        # Forward pass
        outputs = model(features)                     # Dict[str, (B, T, C)]
        
        # Compute loss
        loss = criterion(outputs, components)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = evaluate(model, val_loader)
    
    # Save best model
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'model_best.pt')
```

### 7.9 Evolução Esperada da Loss

```
Epoch 1:  Train Loss: 4.5   Val Loss: 4.2
Epoch 5:  Train Loss: 2.8   Val Loss: 2.9
Epoch 10: Train Loss: 2.0   Val Loss: 2.2
Epoch 20: Train Loss: 1.5   Val Loss: 1.8
Epoch 50: Train Loss: 0.8   Val Loss: 1.2
```

A loss está na escala de ~2.0 porque é a soma de 9 componentes (média ~0.22 por componente).

---

## 8. Inferência e Reassemblagem

### 8.1 Fluxo de Inferência

```
┌─────────────────────────────────────────────────────────────────┐
│                         Inferência                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Áudio → CQT Features → Modelo → 9 Outputs                      │
│                                                                  │
│  Para cada frame t:                                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  root_logits[t]  → softmax → argmax → 'C'               │    │
│  │  bass_logits[t]  → softmax → argmax → 'N'               │    │
│  │  triad_logits[t] → softmax → argmax → 'maj'             │    │
│  │  misc_logits[t]  → softmax → argmax → 'N'               │    │
│  │  7th_logits[t]   → softmax → argmax → '7'               │    │
│  │  9th_logits[t]   → softmax → argmax → 'N'               │    │
│  │  11th_logits[t]  → softmax → argmax → 'N'               │    │
│  │  13th_logits[t]  → softmax → argmax → 'N'               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Reassembler                           │    │
│  │                                                          │    │
│  │  {root:'C', triad:'maj', 7th:'7', ...} → 'C:maj7'       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Lógica de Reassemblagem

```python
class ChordReassembler:
    def reassemble(self, components: Dict[str, str]) -> str:
        root = components['root']
        bass = components['bass']
        triad = components['triad']
        misc = components['misc']
        seventh = components['7th']
        ninth = components['9th']
        eleventh = components['11th']
        thirteenth = components['13th']
        
        # Regra 1: Se root é N, retorna N
        if root == 'N':
            return 'N'
        
        # Regra 2: Power chord
        if misc == '5':
            chord = f"{root}:5"
        
        # Regra 3: Constrói acorde normal
        else:
            # Tríade
            if triad == 'maj':
                chord = root  # C:maj → C
            else:
                chord = f"{root}:{triad}"  # C:min → C:min
            
            # Extensões
            if seventh != 'N':
                chord += '7'  # ou maj7, dependendo
            if ninth != 'N':
                chord += f"({ninth})"
            # ... etc
        
        # Regra 4: Adiciona baixo se diferente
        if bass != 'N' and bass != root:
            chord += f"/{bass}"
        
        return chord
```

### 8.3 Exemplos de Reassemblagem

| Componentes | Acorde Reassemblado |
|-------------|---------------------|
| root=C, triad=maj, 7th=7 | `C:maj7` |
| root=G, triad=min, 7th=b7 | `G:min7` |
| root=D, triad=maj, 7th=b7 | `D:7` |
| root=A, triad=min, bass=E | `A:min/E` |
| root=E, misc=5 | `E:5` |
| root=N | `N` |

### 8.4 Scripts de Inferência

#### Inferência em Áudio Completo

O script `infer_full_audio.py` processa um arquivo de áudio completo e reconhece acordes.
O backbone é detectado automaticamente a partir do checkpoint (`--backbone auto`):

```bash
# Mostrar mudanças de acordes (detecção automática de backbone)
python infer_full_audio.py \
    --config run_config.yaml \
    --checkpoint checkpoints/cf_gradnorm_a15_k0/model_best.pt \
    --audio_file musica.mp3 \
    --device cuda

# Forçar backbone ChordFormer
python infer_full_audio.py \
    --config run_config.yaml \
    --checkpoint checkpoints/cf_gradnorm_a15_k0/model_best.pt \
    --audio_file musica.mp3 \
    --backbone chordformer

# Salvar resultado em arquivo .lab
python infer_full_audio.py \
    --config run_config.yaml \
    --checkpoint checkpoints/cf_gradnorm_a15_k0/model_best.pt \
    --audio_file musica.mp3 \
    --output resultado.lab

# Mostrar TODOS os frames (primeiros 500)
python infer_full_audio.py \
    --config run_config.yaml \
    --checkpoint checkpoints/cf_gradnorm_a15_k0/model_best.pt \
    --audio_file musica.mp3 \
    --show_all
```

**Parâmetros:**

| Parâmetro | Descrição |
|-----------|-----------|
| `--config` | Arquivo de configuração (default: run_config.yaml) |
| `--checkpoint` | Caminho para o modelo treinado (.pt) |
| `--audio_file` | Arquivo de áudio (mp3, wav, etc.) |
| `--backbone` | `auto` (detecta do checkpoint) ou `chordformer` |
| `--device` | cuda ou cpu |
| `--show_all` | Mostra todos os frames, não apenas mudanças |
| `--max_frames` | Limite de frames a exibir (default: 500) |
| `--output` | Salvar resultado em arquivo .lab |

**Exemplo de Output:**

```
=== Chord Changes ===
    0.00s -     3.24s: C:maj
    3.24s -     6.48s: G:min7
    6.48s -    10.02s: F:maj
   10.02s -    13.26s: C:maj7

=== Chord Statistics ===
  C:maj          :   120 frames ( 25.0%)
  G:min7         :    95 frames ( 19.8%)
  F:maj          :    88 frames ( 18.3%)
  ...
```

**Formato do Arquivo .lab:**

```
0.000	3.240	C:maj
3.240	6.480	G:min7
6.480	10.020	F:maj
```

#### Uso Programático (Python)

```python
import torch
from models.btc_model_decomposed import ChordFormer_model_decomposed
from utils.chord_decomposition import ChordReassembler, COMPONENT_NAMES
from utils.hparams import HParams
import librosa
import numpy as np

# Carregar modelo
config = HParams.load('run_config.yaml')
checkpoint = torch.load('checkpoints/cf_gradnorm_a15_k0/model_best.pt', map_location='cpu')
model = ChordFormer_model_decomposed(config=config)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# Carregar áudio e extrair features
y, sr = librosa.load('musica.mp3', sr=22050)
cqt = librosa.cqt(y, sr=sr, n_bins=252, bins_per_octave=36, hop_length=2048)
feature = np.log(np.abs(cqt) + 1e-6)

# Preparar input (primeiro chunk de 108 frames)
chunk = feature[:, :108].T  # (108, 252)
x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)  # (1, 108, 252)

# Inferência
with torch.no_grad():
    output = model(x)
    predictions = output[0] if isinstance(output, tuple) else model.decomposer.get_predictions(output)

# Reassemblar acordes
reassembler = ChordReassembler()
for t in range(10):  # Primeiros 10 frames
    indices = {comp: predictions[comp][0, t].item() for comp in COMPONENT_NAMES}
    chord = reassembler.reassemble_from_indices(indices)
    time_sec = t * 2048 / 22050
    print(f"{time_sec:.2f}s: {chord}")
```

### 8.5 HarmonicCRF: Decodificação Temporal com CRF

O **HarmonicCRF** é um módulo opcional de pós-processamento que aplica um Campo Aleatório Condicional (CRF) sobre as predições de root e triad para suavizar a sequência temporal, evitando transições de acordes implausíveis.

#### Conceito

Em vez de fazer argmax frame a frame (que pode oscilar entre acordes semelhantes), o CRF combina os logits de root (13 classes) e triad (7 classes) em 91 tags conjuntas (root × triad) e aplica o algoritmo de Viterbi para encontrar a sequência globalmente ótima.

```
ChordFormer (congelado) → 9 head logits
    → HarmonicCRF:
        1. Potencial de observação: log P(root) + log P(triad) = (B, T, 91)
        2. Viterbi com transições aprendidas (91×91)
        3. Decodifica tag → root + triad
    → Extensões (bass, 7th, 9th, etc.) resolvidas por argmax
    → ChordReassembler → acorde final
```

A matriz de transição (~8k parâmetros) é treinada em cima dos logits do ChordFormer congelado, aprendendo quais progressões harmônicas o modelo tende a produzir (ex: `C:maj → G:7` é comum, `C:maj → C#:dim` é raro).

#### Treino do HarmonicCRF

Após treinar o ChordFormer, treina-se o CRF separadamente (~minutos):

```bash
python train_harmonic_crf.py \
    --checkpoint checkpoints/meu_run/model_best.pt \
    --config run_config.yaml \
    --train_datasets billboard queen robbiewilliams rwc jaah dj_avan_songbook2 \
    --crf_run_name harmonic_crf_BiQuRoRwJaDj2 \
    --num_epochs 50
```

O script:
1. Carrega o ChordFormer e congela todos os parâmetros
2. Instancia o HarmonicCRF (91 tags, ~8k params treináveis)
3. Loop de treino: logits congelados → CRF loss (NLL) → backprop só nos params do CRF
4. Loga accuracy do CRF vs argmax (baseline) para medir o ganho
5. Early stopping, salva `crf_best.pt`

| Parâmetro | Default | Descrição |
|---|---|---|
| `--checkpoint` | (obrigatório) | Checkpoint do ChordFormer treinado |
| `--train_datasets` | config | Datasets para treino/validação |
| `--num_epochs` | 50 | Épocas de treino do CRF |
| `--learning_rate` | 0.01 | LR para parâmetros do CRF |
| `--crf_run_name` | auto | Nome da run |
| `--early_stop_patience` | 10 | Paciência para early stopping |

#### Inferência com HarmonicCRF

```bash
python run_inference_batch_decomposed.py \
    --checkpoint checkpoints/meu_run/model_best.pt \
    --harmonic_crf checkpoints/harmonic_crf_BiQuRoRwJaDj2/crf_best.pt \
    --test_dataset dj_avan_songbook1 \
    --backbone chordformer
```

O flag `--harmonic_crf` ativa a decodificação Viterbi. Sem o flag, o comportamento é idêntico ao anterior (argmax por frame).

#### Arquivos

- `models/harmonic_crf.py` — Módulo `HarmonicCRF` (potencial de observação, CRF, Viterbi, loss)
- `models/crf_model.py` — CRF core reutilizado (transições, forward algorithm, Viterbi) — **legado, não modificado**
- `train_harmonic_crf.py` — Script de treino do CRF

#### Extensibilidade futura

- Expandir para root × triad × 7th (91 × 4 = 364 tags)
- Adicionar outros heads ao potencial de observação (bass, 7th)
- Treinar end-to-end (descongelar ChordFormer)
- Substituir transições aprendidas por penalidade fixa (comparação com baseline)

---

### 8.6 Avaliação em Batch: Inferência + Métricas

Para avaliar o modelo treinado em um dataset de teste completo e gerar relatórios CSV com métricas, o fluxo é dividido em dois passos:

```
┌─────────────────┐     ┌──────────────────────────────┐     ┌──────────────────────┐
│  Checkpoint     │────▶│ run_inference_batch_          │────▶│  Pasta com .lab      │
│  (model_best.pt)│     │ decomposed.py                │     │  preditos            │
└─────────────────┘     └──────────────────────────────┘     └──────────┬───────────┘
                                                                        │
┌─────────────────┐     ┌──────────────────────────────┐                │
│  Ground Truth   │────▶│ generate_metrics_csv.py       │◀───────────────┘
│  (.lab anotados)│     └──────────────────────────────┘
└─────────────────┘                    │
                                       ▼
                          ┌──────────────────────┐
                          │  metrics_per_track.csv│
                          │  metrics_summary.csv  │
                          └──────────────────────┘
```

#### Passo 1: Inferência Batch (`run_inference_batch_decomposed.py`)

Processa todos os áudios de um dataset e gera arquivos `.lab` com as predições.

**Com `--test_dataset` (resolve o caminho automaticamente do config):**

```bash
python run_inference_batch_decomposed.py \
    --checkpoint /caminho/para/checkpoints/model_best.pt \
    --test_dataset rwc \
    --backbone chordformer
```

**Com `--audio_dir` (caminho explícito para os áudios):**

```bash
python run_inference_batch_decomposed.py \
    --checkpoint /caminho/para/checkpoints/model_best.pt \
    --audio_dir /home/daniel.melo/datasets/rwc/audio \
    --output_dir ./inferences_decomposed/chordformer_rwc \
    --backbone chordformer
```

**Parâmetros:**

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `--checkpoint` | (obrigatório) | Caminho do `.pt` do modelo treinado |
| `--test_dataset` | — | Nome do dataset (`rwc`, `dj_avan`, `billboard`, `jaah`, `queen`, `robbiewilliams`). Resolve `audio_dir` do config |
| `--audio_dir` | — | Alternativa: caminho direto para a pasta de áudios. Mutuamente exclusivo com `--test_dataset` |
| `--backbone` | `chordformer` | Backbone do modelo: `chordformer`, `btc`, ou `auto` (detecta do checkpoint) |
| `--config` | `run_config.yaml` | Arquivo de configuração |
| `--output_dir` | (auto) | Pasta de saída dos `.lab`. Se omitido, gera em `./inferences_decomposed/inference_<exp>_test_<ds>/` |
| `--output_base` | `./inferences_decomposed` | Base para nomes de pasta auto-gerados |
| `--exp_name` | (do checkpoint) | Nome do experimento para a pasta de saída |
| `--device` | `cuda` | Dispositivo (`cuda`, `cpu`) |

**Saída:** Uma pasta com um arquivo `.lab` por música, no formato:

```
0.000 3.240 C:maj
3.240 6.480 G:min7
6.480 10.020 F:maj
```

#### Passo 2: Gerar Métricas (`generate_metrics_csv.py`)

Compara os `.lab` preditos contra os `.lab` de referência (ground truth) do dataset de teste usando `mir_eval`.

```bash
python generate_metrics_csv.py \
    --inference_dir ./inferences_decomposed/chordformer_rwc \
    --gt_dir /home/daniel.melo/datasets/rwc/annotations
```

**Parâmetros:**

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `--inference_dir` | (obrigatório) | Pasta com os `.lab` preditos |
| `--gt_dir` | (obrigatório) | Pasta com os `.lab` de ground truth |
| `--output_dir` | `./metrics_results` | Onde salvar os CSVs |
| `--prefix` | `metrics` | Prefixo dos nomes dos CSVs |

**Saída:** Dois CSVs em `--output_dir`:

| Arquivo | Conteúdo |
|---------|----------|
| `{prefix}_per_track.csv` | Métricas por música: root, majmin, thirds, triads, tetrads, sevenths, mirex, segmentation, etc. |
| `{prefix}_summary.csv` | Médias, desvios padrão e WCSR (Weighted Chord Symbol Recall) de cada métrica |

**Métricas calculadas (via `mir_eval.chord.evaluate`):**

| Métrica | Descrição |
|---------|-----------|
| root | Acurácia da nota fundamental |
| majmin | Acurácia maior/menor |
| thirds | Acurácia considerando terças |
| triads | Acurácia de tríades completas |
| tetrads | Acurácia de tétrades (com 7ª) |
| sevenths | Acurácia da sétima |
| mirex | Score MIREX ACE |
| overseg / underseg / seg | Métricas de segmentação |
| WCSR (por métrica) | Weighted Chord Symbol Recall — pondera pela duração de cada música |

#### Exemplo Completo

```bash
cd /home/daniel.melo/BTC_ORIGINAL/chords_recog/BTC-ISMIR19

# Passo 1: Inferência no RWC
python run_inference_batch_decomposed.py \
    --checkpoint /caminho/checkpoints/model_best.pt \
    --test_dataset rwc \
    --backbone chordformer \
    --output_dir ./inferences_decomposed/chordformer_rwc

# Passo 2: Métricas
python generate_metrics_csv.py \
    --inference_dir ./inferences_decomposed/chordformer_rwc \
    --gt_dir /home/daniel.melo/datasets/rwc/annotations \
    --prefix chordformer_rwc
```

Para testar no Djavan, basta trocar os caminhos:

```bash
python run_inference_batch_decomposed.py \
    --checkpoint /caminho/checkpoints/model_best.pt \
    --test_dataset dj_avan \
    --backbone chordformer \
    --output_dir ./inferences_decomposed/chordformer_djavan

python generate_metrics_csv.py \
    --inference_dir ./inferences_decomposed/chordformer_djavan \
    --gt_dir /home/daniel.melo/datasets/dj_avan/annotations \
    --prefix chordformer_djavan
```

O `--prefix` diferencia os CSVs para não sobrescrever resultados anteriores.

---

## 9. Métricas de Avaliação

### 9.1 Métricas por Componente

| Métrica | Descrição |
|---------|-----------|
| `acc_root` | % de frames com root correto |
| `acc_bass` | % de frames com bass correto |
| `acc_triad` | % de frames com tríade correta |
| `acc_7th` | % de frames com 7ª correta |
| `f1_[comp]` | F1-score macro para cada componente |

### 9.2 Métricas Agregadas

| Métrica | Descrição |
|---------|-----------|
| `acc_component_avg` | Média das 9 acurácias de componentes |
| `acc_root_triad` | % com root E triad corretos |
| `acc_root_triad_7th` | % com root, triad E 7th corretos |
| `acc_full_chord` | % com TODOS os 9 componentes corretos |

### 9.3 Interpretação

```
============================================================
EVALUATION METRICS SUMMARY
============================================================

Total frames evaluated: 150,000

--- Per-Component Accuracy ---
  root    : Acc= 85.2%  F1= 78.5%
  bass    : Acc= 92.1%  F1= 65.3%
  triad   : Acc= 82.4%  F1= 71.2%
  misc    : Acc= 99.1%  F1= 85.0%
  7th     : Acc= 88.5%  F1= 72.8%
  9th     : Acc= 95.2%  F1= 58.4%
  11th    : Acc= 97.8%  F1= 45.2%
  13th    : Acc= 98.5%  F1= 42.1%

--- Aggregate Metrics ---
  Component Avg Accuracy: 92.4%
  Root+Triad Accuracy:    78.5%
  Root+Triad+7th Acc:     72.1%
  Full Chord Accuracy:    65.3%
============================================================
```

**Interpretação:**
- **Root 85%**: O modelo identifica a nota fundamental corretamente em 85% dos frames
- **Triad 82%**: Distingue bem entre maj/min/dim/aug
- **9th/11th/13th baixo F1**: Classes raras, difíceis de prever, mas alta acurácia (maioria é 'N')
- **Full Chord 65%**: Quando exigimos todos os 9 componentes corretos simultaneamente

---

## 10. Debug e Testes

### 10.1 Script de Debug (`debug_model.py`)

O script `debug_model.py` fornece utilidades para testar e debugar o modelo em diferentes níveis.

#### Comandos Básicos

```bash
# Rodar todos os testes
python debug_model.py --config run_config.yaml

# Com checkpoint carregado
python debug_model.py --config run_config.yaml --checkpoint checkpoints/model_best.pt

# Usar CPU se não tiver GPU
python debug_model.py --config run_config.yaml --device cpu
```

#### Testes Disponíveis

| Teste | Comando | Descrição |
|-------|---------|-----------|
| `decompose` | `--test decompose` | Testa decomposição e reassembly de acordes |
| `forward` | `--test forward` | Testa forward pass com dados sintéticos |
| `gradient` | `--test gradient` | Verifica fluxo de gradientes |
| `distribution` | `--test distribution` | Analisa distribuição de classes no dataset |
| `predict` | `--test predict` | Mostra predições decodificadas (requer checkpoint) |
| `all` | `--test all` | Roda todos os testes (padrão) |

```bash
# Exemplos
python debug_model.py --test forward
python debug_model.py --test gradient
python debug_model.py --test distribution --num_samples 500
python debug_model.py --test predict --checkpoint checkpoints/model_best.pt
```

### 10.2 Teste de Forward Pass

Verifica se o modelo processa corretamente dados sintéticos:

```
======================================================================
TEST: Forward Pass with Synthetic Data
======================================================================
Input shape: torch.Size([2, 1, 252, 108])
Expected: (batch=2, 1, features=252, seq_len=108)

--- Predictions ---
  root  : shape=[2, 108], unique=[0, 1, 2, 3, 4]...
  bass  : shape=[2, 108], unique=[0, 1, 2]...
  triad : shape=[2, 108], unique=[0, 1, 2, 3]...
  ...

--- Loss ---
  Total Loss: 2.3456

--- Component Losses ---
  root  : 0.2134 ██
  bass  : 0.1823 █
  triad : 0.3521 ███
  misc  : 0.0812 
  7th   : 0.4234 ████
  9th   : 0.2567 ██
  11th  : 0.1834 █
  13th  : 0.1531 █

✓ Forward pass successful!
```

### 10.3 Teste de Gradientes

Verifica se os gradientes fluem corretamente para todos os componentes:

```
======================================================================
TEST: Gradient Flow Verification
======================================================================
Loss: 2.3456

--- Gradient Norms by Module ---
  conformer_encoder.input_projection   : avg=0.012345, max=0.023456 ✓
  conformer_encoder.conformer_blocks   : avg=0.008234, max=0.015678 ✓
  decomposer.heads                     : avg=0.005678, max=0.009876 ✓

--- Output Head Gradients ---
  root  : 0.005432
  bass  : 0.004567
  triad : 0.006789
  ...

✓ Gradient flow verification complete!
```

### 10.4 Análise de Distribuição

Analisa a distribuição de classes no dataset para identificar desbalanceamentos:

```
======================================================================
TEST: Component Distribution Analysis
======================================================================
Dataset size: 17736
Analyzing 100 samples...

--- Distribution per Component ---

root (vocab size: 13, total frames: 10800):
  C     :   2345 ( 21.7%) ██████████
  G     :   1876 ( 17.4%) ████████
  D     :   1543 ( 14.3%) ███████
  A     :   1234 ( 11.4%) █████
  E     :    987 (  9.1%) ████
  ... and 8 more classes

7th (vocab size: 4, total frames: 10800):
  N     :   8765 ( 81.2%) ████████████████████████████████████████
  7     :   1234 ( 11.4%) █████
  b7    :    567 (  5.2%) ██
  bb7   :    234 (  2.2%) █

✓ Distribution analysis complete!
```

### 10.5 Teste de Decomposição/Reassembly

Verifica se a decomposição e reassembly de acordes funciona corretamente:

```
======================================================================
TEST: Chord Decomposition and Reassembly
======================================================================

--- Decomposition Test ---

  Input: C:maj
    Components: {'root': 'C', 'bass': 'N', 'triad': 'maj', ...}
    Indices: {'root': 1, 'bass': 0, 'triad': 1, ...}
    Reassembled: C:maj ✓

  Input: D:min7
    Components: {'root': 'D', 'bass': 'N', 'triad': 'min', '7th': '7', ...}
    Indices: {'root': 3, 'bass': 0, 'triad': 2, '7th': 1, ...}
    Reassembled: D:min7 ✓

  Input: N
    Components: {'root': 'N', 'bass': 'N', 'triad': 'N', ...}
    Reassembled: N ✓

✓ Decomposition/reassembly test complete!
```

### 10.6 Visualização de Losses por Componente

Durante o treinamento, as losses são exibidas por componente:

```
=== Epoch 7/100 ===
Train Loss: 2.0400
  Components: root:0.210 | bass:0.183 | tria:0.352 | misc:0.081 | 7th:0.423 | 9th:0.256 | 11th:0.183 | 13th:0.153
Val Loss: 2.3280
  Val Components: root:0.245 | bass:0.198 | tria:0.378 | misc:0.092 | 7th:0.456 | 9th:0.278 | 11th:0.198 | 13th:0.167
Saved best checkpoint to checkpoints/my_run/model_best.pt
```

Isso permite identificar:
- Quais componentes estão convergindo mais rápido
- Quais componentes estão estagnados
- Se há overfitting em componentes específicos

---

## Anexo A: Nota sobre PyTorch 2.6+ e `weights_only`

A partir do PyTorch 2.6, o default de `torch.load()` mudou para `weights_only=True`, que bloqueia a deserialização de numpy arrays nos `.pt`. Todos os scripts que leem `.pt` com arrays numpy devem usar:

```python
data = torch.load(path, map_location='cpu', weights_only=False)
```

Scripts já corrigidos: `preprocess_decomposed.py`, `add_original_labels.py`, `train_decomposed.py`, `train_harmonic_crf.py`.

---

## Anexo B: Estrutura de Arquivos

```
BTC-ISMIR19/
├── data/
│   ├── audio_dataset.py              # Dataset base (k-fold, augment)
│   ├── audio_dataset_structured.py   # Dataset com decomposição (9 componentes)
│   └── curriculum_learning.py        # Curriculum learning
├── models/
│   ├── btc_model_decomposed.py       # ChordFormer decomposed + MultiTaskLoss + GradNorm
│   ├── harmonic_crf.py               # HarmonicCRF — CRF root×triad para decodificação temporal
│   ├── crf_model.py                  # CRF core (transições, Viterbi, forward alg.) — legado, reutilizado
│   ├── btc_model.py                  # Modelos legado (BTC, baselines)
│   └── baseline_models.py            # CNN, CRNN, Crf wrapper (legado)
├── utils/
│   ├── chord_decomposition.py        # ChordDecomposer & Reassembler (9 componentes)
│   ├── decomposed_inference.py       # DecomposedChordTrainer, Inference, Metrics, GradNorm update
│   ├── transformer_modules.py        # ConformerEncoder, output layers
│   ├── mir_eval_modules.py           # idx2voca_chord, scoring
│   ├── preprocess.py                 # Feature extraction (CQT)
│   ├── hparams.py                    # HParams (YAML config loader)
│   └── chords.py                     # Chord vocab e intervalos
├── scripts/
│   ├── preprocess_datasets.py        # Preprocessing principal
│   ├── preprocess_decomposed.py      # Preprocessing decomposed
│   ├── add_original_labels.py        # Adiciona labels originais aos .pt
│   ├── compute_normalization.py                # Calcula mean/std global para normalização
│   ├── precompute_class_weights_decomposed.py  # Cache de class weights
│   ├── diagnose_decomposition_mismatch.py      # Diagnóstico train/val
│   └── convert_to_decomposed.py      # Converte .pt 170-class → decomposed
├── train_decomposed.py               # Treino ChordMax (ChordFormer + GradNorm + wandb)
├── train_harmonic_crf.py             # Treino do HarmonicCRF (CRF sobre ChordFormer congelado)
├── train_curriculum.py               # Treino legado
├── infer_decomposed.py               # Inferência janela única
├── infer_full_audio.py               # Inferência áudio completo (chunks)
├── run_inference_batch_decomposed.py # Inferência batch → .lab (suporta --harmonic_crf)
├── generate_metrics_csv.py           # Compara .lab preditos vs ground truth → CSVs de métricas
├── debug_model.py                    # Debug e testes do modelo
├── quick_test_decomposed.py          # Smoke test rápido
└── run_config.yaml                   # Configurações centrais
```

---

## Anexo C: Fluxo Completo — Do Treino à Avaliação

Exemplo prático de ponta a ponta, treinando nos datasets billboard, queen, robbiewilliams, rwc, jaah e dj_avan_songbook2, testando no dj_avan_songbook1.

### Passo 1: Treinar o ChordFormer

```bash
cd /home/daniel.melo/BTC_ORIGINAL/chords_recog/BTC-ISMIR19

python train_decomposed.py \
    --config run_config.yaml \
    --backbone chordformer \
    --kfold 0 \
    --run_name BiQuRoRwJaDj2_8heads_testDj1_k0 \
    --num_epochs 100 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --weight_decay 0.001 \
    --no_class_weights \
    --use_gradnorm \
    --gradnorm_alpha 0.7 \
    --gradnorm_lr 0.005 \
    --wandb_project runs \
    --train_datasets billboard queen robbiewilliams rwc jaah dj_avan_songbook2
```

Resultado: `checkpoints/BiQuRoRwJaDj2_8heads_testDj1_k0/model_best.pt`

### Passo 2: Treinar o HarmonicCRF

Após o ChordFormer terminar, treinar o CRF em cima dos logits congelados (~minutos):

```bash
python train_harmonic_crf.py \
    --checkpoint checkpoints/BiQuRoRwJaDj2_8heads_testDj1_k0/model_best.pt \
    --config run_config.yaml \
    --train_datasets billboard queen robbiewilliams rwc jaah dj_avan_songbook2 \
    --crf_run_name harmonic_crf_BiQuRoRwJaDj2 \
    --num_epochs 50 \
    --learning_rate 0.01
```

O script loga a cada época a accuracy do CRF vs argmax (baseline), mostrando o ganho:

```
Epoch 15/50 | Train loss: 2.31 acc: 0.82 | Val loss: 2.45
  CRF     root: 0.8534  triad: 0.7892  both: 0.7123
  Argmax  root: 0.8401  triad: 0.7756  both: 0.6945
  Delta   root: +0.0133  triad: +0.0136  both: +0.0178
```

Resultado: `checkpoints/harmonic_crf_BiQuRoRwJaDj2/crf_best.pt`

### Passo 3: Inferência no dataset de teste

Rodar duas inferências (com e sem CRF) para comparar o impacto:

```bash
# SEM CRF (baseline — argmax por frame)
python run_inference_batch_decomposed.py \
    --checkpoint checkpoints/BiQuRoRwJaDj2_8heads_testDj1_k0/model_best.pt \
    --test_dataset dj_avan_songbook1 \
    --backbone chordformer \
    --exp_name BiQuRoRwJaDj2_no_crf

# COM CRF (Viterbi temporal)
python run_inference_batch_decomposed.py \
    --checkpoint checkpoints/BiQuRoRwJaDj2_8heads_testDj1_k0/model_best.pt \
    --harmonic_crf checkpoints/harmonic_crf_BiQuRoRwJaDj2/crf_best.pt \
    --test_dataset dj_avan_songbook1 \
    --backbone chordformer \
    --exp_name BiQuRoRwJaDj2_with_crf
```

Resultado: duas pastas em `inferences_decomposed/` com arquivos `.lab`.

### Passo 4: Gerar métricas comparativas

```bash
# Métricas SEM CRF
python generate_metrics_csv.py \
    --inference_dir ./inferences_decomposed/inference_BiQuRoRwJaDj2_no_crf_test_Dj1 \
    --gt_dir /home/daniel.melo/datasets/dj_avan_songbook1/annotations \
    --prefix no_crf_Dj1

# Métricas COM CRF
python generate_metrics_csv.py \
    --inference_dir ./inferences_decomposed/inference_BiQuRoRwJaDj2_with_crf_test_Dj1 \
    --gt_dir /home/daniel.melo/datasets/dj_avan_songbook1/annotations \
    --prefix with_crf_Dj1
```

Resultado: CSVs em `metrics_results/` com métricas por track e agregadas (WCSR, root, triads, etc.).

### Passo 5: Análise visual

Subir o visualizador para analisar as predições qualitativamente:

```bash
ssh -i ~/chave_gcp -L 8050:localhost:8050 daniel.melo@34.55.222.142
# Na VM:
cd /home/daniel.melo/BTC_ORIGINAL/chords_recog
python -m visualizer
```

No browser (`http://localhost:8050`):
1. Selecionar o dataset `dj_avan_songbook1`
2. Alternar entre as pastas de inferência (com/sem CRF) no dropdown
3. Comparar os erros no timeline e na tabela de segmentos
4. Usar a Chord Search (`/search`) para encontrar acordes específicos nos dados GT

---

## Referências

1. **ChordMax (ChordFormer)**: Conformer encoder + decomposição multi-tarefa em 9 componentes
3. **GradNorm: Gradient Normalization for Adaptive Loss Balancing** (ICML 2018) - Balanceamento adaptativo de tarefas
4. **mir_eval** - Biblioteca padrão para avaliação MIR

---

**Última atualização:** Março 2026
