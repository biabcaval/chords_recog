# 🎯 Priorização de Melhorias para o BTC

Este documento analisa o estado atual do modelo BTC e prioriza as melhorias baseadas no ChordFormer por impacto vs esforço de implementação.

---

## Análise do Estado Atual

O modelo BTC atualmente usa:
- ✅ Cross-entropy loss padrão (`F.nll_loss`)
- ✅ 170 classes (vocabulário grande)
- ❌ **Sem class reweighting** (todas as classes têm peso igual)
- ❌ **Sem CRF na inferência** (apesar de existir `crf_model.py`)
- ❌ **Saída flat** (não estruturada)

---

## 🏆 Ranking de Prioridade (Impacto vs Esforço)

| Prioridade | Melhoria | Impacto | Esforço | Justificativa |
|------------|----------|---------|---------|---------------|
| **1️⃣** | **Reweighted Loss** | 🔥🔥🔥 Alto | 🟢 Baixo | Muda ~10 linhas de código, melhora muito acordes raros |
| **2️⃣** | **CRF Decoding** | 🔥🔥 Médio-Alto | 🟢 Baixo | Já existe no código, só precisa ativar |
| **3️⃣** | **Data Augmentation (Pitch Shift)** | 🔥🔥 Médio | 🟡 Médio | Aumenta diversidade dos dados |
| **4️⃣** | **Saída Estruturada** | 🔥🔥🔥 Alto | 🔴 Alto | Mudança arquitetural significativa |

---

## 1️⃣ MAIS URGENTE: Reweighted Loss

### Por quê?
- **Problema atual:** Acordes como `dim`, `aug`, `hdim7`, `dim7` têm MUITO menos exemplos que `maj` e `min`
- O modelo tende a prever sempre acordes comuns
- **acc_class** (acurácia por classe) provavelmente está baixa

### Impacto esperado:
- ChordFormer reporta **+10% em acc_class** com reweighting
- Melhora significativa em acordes raros sem perder muito em acordes comuns

### Esforço:
- **~10-15 linhas de código** para implementar
- Não muda arquitetura, só a função de loss

### Implementação rápida:

```python
# Em train_curriculum.py ou transformer_modules.py

def compute_class_weights(train_dataset, num_classes=170, gamma=0.5, w_max=10.0):
    """Calcula pesos inversamente proporcionais à frequência"""
    class_counts = torch.zeros(num_classes)
    for data in train_dataset:
        chords = data['chord']
        for c in chords:
            class_counts[c] += 1
    
    # Evita divisão por zero
    class_counts = torch.clamp(class_counts, min=1)
    
    # Peso inversamente proporcional à frequência
    N_max = class_counts.max()
    weights = torch.clamp((N_max / class_counts) ** gamma, max=w_max)
    
    return weights

# Na função de loss:
# Antes:  F.nll_loss(log_probs, labels)
# Depois: F.nll_loss(log_probs, labels, weight=class_weights)
```

### Hiperparâmetros recomendados (do ChordFormer):

| gamma | w_max | Efeito |
|-------|-------|--------|
| 0.3 | 10.0 | Leve rebalanceamento |
| 0.5 | 10.0 | Moderado (recomendado para começar) |
| 0.7 | 20.0 | Forte rebalanceamento |
| 1.0 | 20.0 | Muito forte (pode prejudicar acordes comuns) |

---

## 2️⃣ Segunda Prioridade: CRF Decoding

### Por quê?
- Você já tem `crf_model.py` implementado!
- Reduz transições "impossíveis" entre acordes
- Suaviza predições ruidosas

### Impacto esperado:
- Melhora **segmentação** (menos fragmentação)
- Predições mais coerentes musicalmente

### Esforço:
- Já existe no código, só precisa integrar na inferência

### Como ativar:

```python
# No run_config.yaml, já existe:
model:
  probs_out: False  # Mudar para True para usar CRF

# No treinamento, usar train_crf.py que já existe
```

---

## 3️⃣ Terceira Prioridade: Data Augmentation (Pitch Shift)

### Por quê?
- ChordFormer usa pitch shift de -5 a +6 semitons
- Aumenta diversidade sem coletar mais dados
- Cada música gera 12 versões (original + 11 transposições)

### Impacto esperado:
- Melhor generalização
- Mais robusto a variações de tonalidade
- ~12x mais dados de treinamento

### Esforço:
- Modificar `audio_dataset.py` ou `preprocess_datasets.py`
- Ajustar labels de acordo com a transposição

### Implementação básica:

```python
def pitch_shift_cqt(cqt_features, semitones, bins_per_octave=24):
    """Transpõe CQT por n semitons (roll nos bins)"""
    shift_bins = semitones * (bins_per_octave // 12)
    return np.roll(cqt_features, shift_bins, axis=0)

def transpose_chord_label(chord_id, semitones, num_qualities=14):
    """Transpõe o label do acorde"""
    if chord_id >= 168:  # N ou Unknown
        return chord_id
    root = chord_id // num_qualities
    quality = chord_id % num_qualities
    new_root = (root + semitones) % 12
    return new_root * num_qualities + quality
```

---

## 4️⃣ Última Prioridade: Saída Estruturada

### Por quê deixar por último?
- Requer mudanças significativas no dataset, modelo e treinamento
- Os ganhos podem ser alcançados parcialmente com reweighted loss
- Maior risco de introduzir bugs
- Requer re-treinar do zero

### Quando implementar:
- Depois de validar que as outras melhorias funcionam
- Se precisar escalar para vocabulário ainda maior (>300 acordes)
- Se quiser melhor interpretabilidade dos erros

---

## 📊 Resumo Visual

```
Impacto
   ↑
   │  ┌─────────────────┐
   │  │ Saída           │
   │  │ Estruturada     │
   │  └────────┬────────┘
   │           │
   │  ┌────────┴────────┐
   │  │ Reweighted Loss │  ← COMECE AQUI!
   │  │ (URGENTE)       │
   │  └────────┬────────┘
   │           │
   │  ┌────────┴────────┐
   │  │ CRF Decoding    │
   │  └────────┬────────┘
   │           │
   │  ┌────────┴────────┐
   │  │ Data Augment    │
   │  └─────────────────┘
   │
   └──────────────────────────→ Esforço
        Baixo              Alto
```

---

## 📈 Ganhos Esperados (Baseado no ChordFormer)

| Melhoria | acc_frame | acc_class | Observação |
|----------|-----------|-----------|------------|
| Baseline (sem melhorias) | ~77% | ~33% | Estado atual estimado |
| + Reweighted Loss (0.5, 10) | ~77% | ~44% | **+11% acc_class!** |
| + CRF Decoding | ~78% | ~45% | Melhora segmentação |
| + Data Augmentation | ~79% | ~46% | Melhor generalização |
| + Saída Estruturada | ~80% | ~48% | Ganho incremental |

---

## 🎯 Plano de Ação Recomendado

### Fase 1: Quick Wins (1-2 dias)
1. [ ] Implementar Reweighted Loss
2. [ ] Treinar e comparar métricas (acc_frame vs acc_class)
3. [ ] Ajustar hiperparâmetros (gamma, w_max)

### Fase 2: Integração CRF (1 dia)
1. [ ] Ativar CRF na inferência
2. [ ] Comparar qualidade das predições

### Fase 3: Data Augmentation (2-3 dias)
1. [ ] Implementar pitch shifting no preprocessing
2. [ ] Re-processar datasets
3. [ ] Treinar com dados aumentados

### Fase 4: Saída Estruturada (1-2 semanas)
1. [ ] Redesenhar arquitetura
2. [ ] Modificar dataset
3. [ ] Treinar e validar

---

## 🔧 Arquivos a Modificar por Melhoria

| Melhoria | Arquivos |
|----------|----------|
| Reweighted Loss | `train_curriculum.py`, `utils/transformer_modules.py` |
| CRF Decoding | `test.py`, `run_config.yaml` |
| Data Augmentation | `audio_dataset.py`, `preprocess_datasets.py` |
| Saída Estruturada | `btc_model.py`, `audio_dataset.py`, `train.py`, `test.py`, `utils/chords.py` |

---

**Última atualização:** Dezembro 2024

