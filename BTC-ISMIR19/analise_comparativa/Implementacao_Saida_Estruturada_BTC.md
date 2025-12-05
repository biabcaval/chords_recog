# ✅ Implementação de Saída Estruturada no BTC

Este documento descreve como seria possível modificar o modelo BTC para ter uma saída estruturada similar ao ChordFormer.

---

## 📊 O que precisaria mudar

### Arquitetura Atual (Flat)

```
Encoder (Bi-directional Transformer)
         ↓
    [batch, 108, 128]
         ↓
   Linear(128 → 170)
         ↓
      Softmax
         ↓
   170 classes de acordes
```

### Arquitetura Proposta (Structured)

```
Encoder (Bi-directional Transformer)
         ↓
    [batch, 108, 128]
         ↓
    ┌────┴────┬────────┬────────┬────────┬────────┐
    ↓         ↓        ↓        ↓        ↓        ↓
 Root+Triad  Bass     7th      9th     11th    13th
 Linear(85) Linear(13) Linear(4) Linear(4) Linear(3) Linear(3)
    ↓         ↓        ↓        ↓        ↓        ↓
 Softmax   Softmax  Softmax  Softmax  Softmax  Softmax
```

---

## 🔧 Mudanças Necessárias

### 1. **Novo modelo com saída estruturada** (`btc_model_structured.py`)

Criar uma nova classe que herda a parte do encoder do BTC mas tem múltiplas cabeças de saída.

### 2. **Modificar o dataset** para retornar labels estruturados

O `audio_dataset.py` precisaria retornar um dicionário com os componentes separados:
```python
{
    'root': tensor([0, 0, 2, 2, ...]),      # 0=C, 1=C#, 2=D, ...
    'quality': tensor([1, 1, 0, 0, ...]),   # 0=min, 1=maj, 2=dim, ...
    'bass': tensor([0, 0, 0, 0, ...]),      # 0=root, 1=2nd, ...
    '7th': tensor([0, 0, 0, 0, ...]),       # 0=N, 1=7, 2=b7, ...
    ...
}
```

### 3. **Modificar o treinamento** para calcular loss de cada componente

```python
total_loss = (
    loss_root * weight_root +
    loss_quality * weight_quality +
    loss_bass * weight_bass +
    loss_7th * weight_7th +
    ...
)
```

### 4. **Modificar a inferência** para combinar os componentes

```python
def reconstruct_chord(root, quality, bass, seventh, ...):
    # Combina os componentes para formar o label do acorde
    chord_label = f"{ROOT_NAMES[root]}:{QUALITY_NAMES[quality]}"
    if bass != 0:
        chord_label += f"/{BASS_NAMES[bass]}"
    if seventh != 0:
        chord_label += f"/{SEVENTH_NAMES[seventh]}"
    return chord_label
```

---

## 📐 Componentes da Saída Estruturada

Baseado no vocabulário atual do BTC (170 acordes = 12 roots × 14 qualities + 2 especiais):

| Componente | Classes | Descrição |
|------------|---------|-----------|
| **Root** | 13 | N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B |
| **Quality** | 15 | N, maj, min, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7, sus2, sus4 |
| **Bass** | 13 | N, 1, 2, b3, 3, 4, 5, 6, b7, 7 (ou notas absolutas) |

**Total: 13 + 15 + 13 = 41 classes** vs **170 classes** (redução de 76%!)

### Opção Expandida (similar ao ChordFormer)

| Componente | Classes | Valores |
|------------|---------|---------|
| **Root + Triad** | 85 | 12 notas × 7 triads + N |
| **Bass** | 13 | N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B |
| **7th** | 4 | N, 7, ♭7, ♭♭7 |
| **9th** | 4 | N, 9, #9, ♭9 |
| **11th** | 3 | N, 11, #11 |
| **13th** | 3 | N, 13, ♭13 |

**Total: 85 + 13 + 4 + 4 + 3 + 3 = 112 classes**

---

## 💡 Vantagens

1. **Menos parâmetros** na camada de saída (41-112 vs 170-301)
2. **Melhor generalização** para acordes raros
3. **Mais interpretável** - você sabe exatamente o que o modelo está prevendo
4. **Facilita análise de erros** - erro no root? na quality? no bass?
5. **Escalável** - adicionar novas extensões não explode o número de classes

---

## ⚠️ Desafios

1. **Precisa re-processar os labels** do dataset
2. **Loss function** precisa combinar as múltiplas perdas com pesos adequados
3. **Inferência** precisa reconstruir o acorde a partir dos componentes
4. **Nem todas as combinações são válidas** (ex: C:maj com bass em D# não faz sentido)
5. **Validação de combinações** pode requerer CRF ou regras de pós-processamento

---

## 🏗️ Estrutura de Arquivos a Criar

```
BTC-ISMIR19/
├── btc_model_structured.py      # Novo modelo com saída estruturada
├── train_structured.py          # Script de treinamento adaptado
├── test_structured.py           # Script de inferência adaptado
└── utils/
    └── chord_components.py      # Funções para converter entre flat e structured
```

---

## 📝 Pseudocódigo do Modelo Estruturado

```python
class StructuredOutputLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Múltiplas cabeças de saída
        self.root_head = nn.Linear(hidden_size, 13)      # 12 notas + N
        self.quality_head = nn.Linear(hidden_size, 15)   # 14 qualities + N
        self.bass_head = nn.Linear(hidden_size, 13)      # 12 notas + N
        # Opcionais para extensões
        self.seventh_head = nn.Linear(hidden_size, 4)    # N, 7, b7, bb7
        self.ninth_head = nn.Linear(hidden_size, 4)      # N, 9, #9, b9
        self.eleventh_head = nn.Linear(hidden_size, 3)   # N, 11, #11
        self.thirteenth_head = nn.Linear(hidden_size, 3) # N, 13, b13
    
    def forward(self, hidden):
        return {
            'root': self.root_head(hidden),
            'quality': self.quality_head(hidden),
            'bass': self.bass_head(hidden),
            'seventh': self.seventh_head(hidden),
            'ninth': self.ninth_head(hidden),
            'eleventh': self.eleventh_head(hidden),
            'thirteenth': self.thirteenth_head(hidden),
        }


class BTC_model_structured(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Encoder igual ao BTC original
        self.self_attn_layers = bi_directional_self_attention_layers(...)
        # Nova camada de saída estruturada
        self.output_layer = StructuredOutputLayer(config['hidden_size'])
    
    def forward(self, x, labels):
        # Encoder
        self_attn_output, weights_list = self.self_attn_layers(x)
        # Saída estruturada
        outputs = self.output_layer(self_attn_output)
        return outputs, weights_list
    
    def compute_loss(self, outputs, labels):
        # Loss para cada componente
        loss_root = F.cross_entropy(outputs['root'], labels['root'])
        loss_quality = F.cross_entropy(outputs['quality'], labels['quality'])
        loss_bass = F.cross_entropy(outputs['bass'], labels['bass'])
        # ... outras losses
        
        # Combinação ponderada
        total_loss = (
            1.0 * loss_root +
            1.0 * loss_quality +
            0.5 * loss_bass +
            0.3 * loss_seventh +
            # ...
        )
        return total_loss
```

---

## 📊 Comparação de Parâmetros

| Modelo | Camada de Saída | Parâmetros (hidden=128) |
|--------|-----------------|-------------------------|
| BTC Original (170 classes) | Linear(128→170) | 21,760 |
| BTC Original (301 classes) | Linear(128→301) | 38,528 |
| BTC Structured (41 classes) | 3 × Linear | 5,248 |
| BTC Structured (112 classes) | 6 × Linear | 14,336 |

**Economia de até 75% nos parâmetros da camada de saída!**

---

## 🎯 Próximos Passos (quando for implementar)

1. [ ] Criar `utils/chord_components.py` com funções de conversão
2. [ ] Criar `btc_model_structured.py` com o novo modelo
3. [ ] Modificar `audio_dataset.py` para retornar labels estruturados
4. [ ] Criar `train_structured.py` com nova função de loss
5. [ ] Criar `test_structured.py` com reconstrução de acordes
6. [ ] Testar e comparar resultados com modelo original

---

**Última atualização:** Dezembro 2024

