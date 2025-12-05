# 📚 Análise do Curriculum Learning no BTC

Este documento analisa a implementação de Curriculum Learning já existente no BTC e sua relação com outras técnicas como Reweighted Loss.

---

## ✅ O que já está implementado (Excelente!)

A implementação em `curriculum_learning.py` é **muito completa**:

### 1. Múltiplas Estratégias de Dificuldade

```python
strategy: 'chord_complexity'    # Acordes simples → complexos
strategy: 'change_frequency'    # Poucas mudanças → muitas mudanças  
strategy: 'unique_chords'       # Poucos acordes únicos → muitos
strategy: 'mixed'               # Combinação ponderada (40%/30%/30%)
```

### 2. Múltiplos Pacings (Ritmo de Progressão)

```python
pacing: 'linear'       # Progressão constante
pacing: 'quadratic'    # Começa devagar, acelera
pacing: 'exponential'  # Começa muito devagar, acelera muito
pacing: 'step'         # 4 degraus discretos
```

### 3. Pesos de Complexidade Bem Definidos

```python
CHORD_COMPLEXITY = {
    'N': 0.5,      # Mais fácil
    'maj': 1.0,    # Fácil
    'min': 1.0,    # Fácil
    '7': 2.0,      # Médio
    'maj7': 2.5,   # Médio-difícil
    'min7': 2.5,   # Médio-difícil
    'dim': 3.0,    # Difícil
    'aug': 3.0,    # Difícil
    'hdim7': 3.5,  # Muito difícil
    'dim7': 3.5,   # Muito difícil
    'X': 4.0,      # Mais difícil (desconhecido)
}
```

### 4. Funcionalidades Implementadas

- `CurriculumLearning`: Classe principal com cálculo de dificuldade
- `CurriculumDataLoader`: DataLoader que ajusta amostras por época
- `get_curriculum_ratio(epoch)`: Calcula % de dados a usar
- `get_sample_indices(epoch)`: Retorna índices ordenados por dificuldade
- `get_stats(epoch)`: Estatísticas para logging/debug

---

## 🤔 Curriculum Learning vs Reweighted Loss

| Aspecto | Curriculum Learning | Reweighted Loss |
|---------|--------------------|-----------------| 
| **O que faz** | Ordena amostras por dificuldade | Dá mais peso a classes raras |
| **Foco** | Facilitar aprendizado inicial | Equilibrar classes |
| **Problema que resolve** | Convergência difícil | Class imbalance |
| **Quando atua** | Seleção de amostras | Cálculo da loss |
| **Complementares?** | ✅ **SIM!** | ✅ **SIM!** |

### 🎯 São técnicas COMPLEMENTARES, não excludentes!

- **Curriculum Learning:** "Aprenda primeiro músicas fáceis, depois difíceis"
- **Reweighted Loss:** "Preste mais atenção em acordes raros"

---

## 📊 O que o ChordFormer menciona sobre Curriculum Learning

O paper do ChordFormer **cita** curriculum learning como trabalho relacionado:

> "Curriculum learning [7] was also adopted to progressively introduce rare chords during training, leveraging hierarchical relationships between base and extended chord qualities to enhance classification performance."

A referência [7] é: **"Curriculum learning for imbalanced classification in large vocabulary automatic chord recognition" (ISMIR 2021)** - que é exatamente o que está implementado no BTC!

**Importante:** O ChordFormer **não implementa** curriculum learning diretamente - eles focaram apenas no reweighted loss. O BTC pode ter **ambos**!

---

## ✅ Veredicto: MANTENHA o Curriculum Learning!

### Por quê?

1. **Já está implementado** e bem feito
2. **Complementa** o reweighted loss
3. **Paper de 2021** (ISMIR) mostra que funciona para chord recognition
4. **Baixo custo** - só precisa ativar no config
5. **Vantagem competitiva** sobre o ChordFormer

---

## ⚙️ Configuração Atual vs Recomendada

### Configuração Atual (Desativado)

```yaml
curriculum:
  enabled: False  # ← DESATIVADO!
  strategy: 'mixed'
  pacing: 'linear'
  start_ratio: 0.3
  pace_epochs: 30
  difficulty_threshold: 0.5
  warmup_epochs: 5
```

### Configuração Recomendada (Ativar!)

```yaml
curriculum:
  enabled: True                    # ← ATIVAR!
  strategy: 'mixed'                # Melhor estratégia (combina 3 métricas)
  pacing: 'linear'                 # Ou 'quadratic' para ser mais conservador
  start_ratio: 0.3                 # Começar com 30% mais fáceis
  pace_epochs: 30                  # Aumentar gradualmente em 30 épocas
  difficulty_threshold: 0.5        # Threshold para fácil/difícil
  warmup_epochs: 5                 # 5 épocas só com amostras fáceis
```

---

## 🏆 Ranking Atualizado de Prioridades

| Prioridade | Melhoria | Status | Ação |
|------------|----------|--------|------|
| **1️⃣** | **Reweighted Loss** | ❌ Não implementado | Implementar |
| **2️⃣** | **Curriculum Learning** | ✅ Implementado | **Ativar!** (`enabled: True`) |
| **3️⃣** | **CRF Decoding** | ✅ Implementado | Integrar na inferência |
| **4️⃣** | **Data Augmentation** | ❌ Não implementado | Implementar depois |
| **5️⃣** | **Saída Estruturada** | ❌ Não implementado | Futuro |

---

## 💡 Combinação Ideal: Curriculum + Reweighted Loss

A melhor estratégia seria usar **AMBOS** simultaneamente:

```
Curriculum Learning + Reweighted Loss
         ↓
Época 1-5:   Só amostras fáceis (30%) + loss reponderada
Época 6-35:  Aumenta dificuldade gradualmente + loss reponderada
Época 36+:   Todos os dados (100%) + loss reponderada
```

### Benefícios da Combinação:

| Fase | Curriculum Learning | Reweighted Loss | Resultado |
|------|--------------------|-----------------| ----------|
| Início | Amostras fáceis | Peso maior em raros | Convergência estável |
| Meio | Aumenta dificuldade | Peso maior em raros | Aprende progressivamente |
| Final | Todos os dados | Peso maior em raros | Generalização balanceada |

---

## 📈 Fluxo de Treinamento com Ambas Técnicas

```
┌─────────────────────────────────────────────────────────────┐
│                    ÉPOCA 1-5 (Warmup)                       │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ 30% amostras    │ →  │ Loss com pesos  │ → Backprop     │
│  │ mais fáceis     │    │ para classes    │                │
│  └─────────────────┘    │ raras           │                │
│                         └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   ÉPOCA 6-35 (Progressão)                   │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ 30% → 100%      │ →  │ Loss com pesos  │ → Backprop     │
│  │ gradualmente    │    │ para classes    │                │
│  └─────────────────┘    │ raras           │                │
│                         └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    ÉPOCA 36+ (Completo)                     │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ 100% amostras   │ →  │ Loss com pesos  │ → Backprop     │
│  │ (todas)         │    │ para classes    │                │
│  └─────────────────┘    │ raras           │                │
│                         └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Resumo

| Pergunta | Resposta |
|----------|----------|
| Curriculum Learning é bom? | ✅ **SIM, muito bom!** |
| Deve manter? | ✅ **SIM, definitivamente!** |
| Está ativado? | ❌ Não (`enabled: False`) |
| O que fazer? | **Ativar** + adicionar **reweighted loss** |
| Vantagem sobre ChordFormer? | ✅ BTC pode ter AMBAS as técnicas |

---

## 🎯 Próximos Passos

1. [ ] Ativar curriculum learning (`enabled: True`)
2. [ ] Implementar reweighted loss
3. [ ] Treinar com ambas as técnicas
4. [ ] Comparar resultados (acc_frame vs acc_class)

---

**Última atualização:** Dezembro 2024

