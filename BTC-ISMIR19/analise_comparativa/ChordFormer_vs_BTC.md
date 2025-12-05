# 🔍 ChordFormer vs BTC - Análise Comparativa

Este documento compara o modelo **BTC (Bi-directional Transformer for Chord Recognition)** com o **ChordFormer**, baseado no paper "ChordFormer: A Conformer-Based Architecture for Large-Vocabulary Audio Chord Recognition" (arXiv:2502.11840v1).

---

## 🎯 Pergunta Principal: Como evitar camada de saída gigantesca?

O ChordFormer resolve isso com **Chord Structure Decomposition** (Decomposição Estrutural de Acordes):

### ❌ Abordagem do BTC (Flat Classification)

```
Saída: 170 classes (vocabulário grande)
       ou 25 classes (maj/min)
       
Camada final: Linear(128 → 170)
```

**Problema:** Uma classe para cada acorde possível = muitas classes!

### ✅ Abordagem do ChordFormer (Structured Representation)

Em vez de prever o acorde inteiro, prevê **6 componentes separados**:

```
Saída: 6 vetores menores

1. Root + Triad: 85 classes (12 notas × 7 tipos + N)
   → {N, C:maj, C:min, C:sus4, C:sus2, C:dim, C:aug, C#:maj, ...}

2. Bass: 13 classes
   → {N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B}

3. 7th: 4 classes
   → {N, 7, ♭7, ♭♭7}

4. 9th: 4 classes
   → {N, 9, #9, ♭9}

5. 11th: 3 classes
   → {N, 11, #11}

6. 13th: 3 classes
   → {N, 13, ♭13}
```

### 📊 Comparação de Parâmetros na Camada de Saída

| Modelo | Classes | Parâmetros (assumindo hidden=256) |
|--------|---------|-----------------------------------|
| **BTC (170 acordes)** | 170 | 256 × 170 = **43,520** |
| **BTC (301 acordes)** | 301 | 256 × 301 = **77,056** |
| **ChordFormer** | 85+13+4+4+3+3 = 112 | 256 × 112 = **28,672** |

**Redução de ~37% nos parâmetros da camada de saída!**

---

## 🏗️ Diferenças Arquiteturais

| Aspecto | BTC | ChordFormer |
|---------|-----|-------------|
| **Arquitetura base** | Transformer Bidirecional | Conformer |
| **Feed-Forward** | Conv1D + Conv1D | FFN + Conv + FFN (Macaron-style) |
| **Atenção** | Bidirecional (forward + backward separados) | Self-attention com positional encoding relativo |
| **Camada de saída** | Flat (170 classes) | **Estruturada (6 componentes)** |
| **Decodificação** | Softmax direto | **CRF (Conditional Random Field)** |
| **Loss** | Cross-entropy | **Reweighted loss** |

---

## 🔧 Detalhes Técnicos do ChordFormer

### 1. Conformer Block

```
Input
  ↓
FFN (½ step) ─────────────────┐
  ↓                           │ Residual
Multi-Head Self-Attention ────┤
  ↓                           │
Convolution Module ───────────┤
  ↓                           │
FFN (½ step) ─────────────────┘
  ↓
Layer Norm
  ↓
Output
```

### 2. Hiperparâmetros do ChordFormer

```yaml
input_dim: 256
num_heads: 4
ffn_dim: 1024
num_layers: 4
depthwise_conv_kernel_size: 31
output_dim: 100
```

### 3. Preprocessing

- **CQT:** 252 bins (36 bins/oitava × 7 oitavas)
- **Sample rate:** 22,050 Hz
- **Hop length:** 512
- **Data augmentation:** Pitch shift de -5 a +6 semitons

---

## ⚖️ Reweighted Loss (Perda Reponderada)

O ChordFormer usa uma função de perda que dá **mais peso para acordes raros**:

```python
# Fórmula simplificada
weight(c) = min(w_max, (N_max / N_c)^γ)

# Onde:
# N_c = frequência da classe c
# N_max = frequência da classe mais comum
# γ = fator de suavização (0.3 a 1.0)
# w_max = peso máximo (10 a 20)
```

Isso resolve o problema de **class imbalance** (acordes raros como `dim7`, `aug` têm poucos exemplos).

---

## 📈 Resultados Comparativos (do paper ChordFormer)

| Métrica | BTC+CNN | CNN+BLSTM | ChordFormer |
|---------|---------|-----------|-------------|
| Root | 54.28% | 83.39% | **84.69%** |
| Thirds | 47.94% | 80.04% | **81.75%** |
| MajMin | 49.00% | 82.62% | **84.09%** |
| Triads | 44.67% | 75.91% | **77.55%** |
| Sevenths | 37.99% | 69.78% | **72.28%** |
| Tetrads | 34.01% | 62.87% | **65.32%** |
| MIREX | 47.94% | 81.52% | **83.62%** |

*(Nota: BTC+CNN no paper é uma versão modificada sem position embedding)*

### Métricas de Large Vocabulary (301 acordes)

| Re-weighting | acc_frame (CNN+BLSTM) | acc_frame (ChordFormer) | acc_class (CNN+BLSTM) | acc_class (ChordFormer) |
|--------------|----------------------|------------------------|----------------------|------------------------|
| No Re-weighting | 0.7676 | **0.7877** | 0.3315 | **0.3884** |
| (0.3, 10.0) | 0.7659 | **0.7801** | 0.3692 | **0.4426** |
| (0.5, 10.0) | 0.7402 | **0.7772** | 0.3821 | **0.4406** |
| (0.7, 20.0) | 0.7117 | **0.7416** | 0.3823 | **0.4471** |
| (1.0, 20.0) | 0.6512 | **0.6994** | 0.3546 | **0.4157** |

---

## 🎯 Resumo das Principais Inovações do ChordFormer

1. **Structured Output:** 6 cabeças de saída menores em vez de 1 gigante
2. **Conformer:** Combina Conv + Attention de forma mais eficiente (Macaron-style)
3. **CRF Decoding:** Suaviza transições entre acordes com penalidade para mudanças
4. **Reweighted Loss:** Equilibra classes raras vs comuns
5. **Relative Positional Encoding:** Melhor generalização temporal (Transformer-XL style)

---

## 💡 Possíveis Melhorias para o BTC

### 1. Structured Output
Dividir a saída em componentes (root, quality, bass, extensions) em vez de classificação flat.

### 2. Reweighted Loss
Implementar perda reponderada para dar mais peso a acordes raros:

```python
def reweighted_loss(logits, labels, class_counts, gamma=0.5, w_max=10.0):
    N_max = class_counts.max()
    weights = torch.clamp((N_max / class_counts) ** gamma, max=w_max)
    return F.cross_entropy(logits, labels, weight=weights)
```

### 3. CRF Decoding
Já existe no código (`crf_model.py`) - usar para suavizar predições.

### 4. Mais Data Augmentation
Adicionar pitch shifting (-5 a +6 semitons) como no ChordFormer.

---

## 📚 Referências

- **BTC Paper:** "A Bi-Directional Transformer for Musical Chord Recognition" (ISMIR 2019)
- **ChordFormer Paper:** "ChordFormer: A Conformer-Based Architecture for Large-Vocabulary Audio Chord Recognition" (arXiv:2502.11840v1, Feb 2025)
- **Conformer Paper:** "Conformer: Convolution-augmented Transformer for Speech Recognition" (Gulati et al., 2020)

---

**Última atualização:** Dezembro 2024

