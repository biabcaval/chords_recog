# 📊 Análise de Requisitos de Dados: BTC vs ChordFormer

Este documento analisa quantas músicas e qual diversidade de acordes é necessária para o BTC superar o ChordFormer, considerando as melhorias propostas.

---

## 🎵 Dataset do ChordFormer

Do paper (Seção IV-A):

```
Dataset: Humphrey and Bello [17], [26]
- Total: 1,217 músicas
- Fontes: Isophonics + Billboard + MARL
- Split: 60% treino / 20% validação / 20% teste
- Vocabulário: 301 acordes distintos
- Data Augmentation: Pitch shift -5 a +6 semitons (12x mais dados)
```

### Cálculo de Dados Efetivos:

```
1,217 músicas × 60% treino = ~730 músicas de treino
730 músicas × 12 (pitch shifts) = ~8,760 músicas efetivas
```

---

## 🎯 O Problema: Class Imbalance

O gráfico do paper mostra a distribuição de acordes:

| Tipo de Acorde | Frequência | % do Dataset |
|----------------|------------|--------------|
| `maj` | ~1200 aparições | ~25% |
| `min` | ~800 aparições | ~17% |
| `7` | ~300 aparições | ~6% |
| `dim`, `aug`, `sus` | ~50-100 cada | ~1-2% cada |
| Extensões (9th, 11th, 13th) | <50 cada | <1% cada |

**Problema:** ~80% dos dados são `maj`, `min` e `7`. Os outros 14 tipos dividem os 20% restantes!

---

## 📐 Quantas Músicas Você Precisa?

### Cenário 1: Sem Melhorias (Baseline)

Para igualar ChordFormer sem reweighted loss:
- **~2,000+ músicas** com boa diversidade
- Ou ~1,200 músicas + data augmentation extensivo

### Cenário 2: Com Reweighted Loss + Curriculum (Recomendado)

Com as melhorias implementadas:
- **~500-800 músicas** com boa diversidade podem ser suficientes
- O reweighted loss compensa a falta de exemplos raros

### Cenário 3: Com Saída Estruturada

Com arquitetura otimizada:
- **~300-500 músicas** podem ser suficientes
- A estrutura reduz a necessidade de exemplos por classe

---

## 🎼 O que é "Diversidade de Acordes Interessante"?

### Distribuição Ideal por Música:

| Tipo | % Ideal | Músicas com esse tipo |
|------|---------|----------------------|
| `maj` / `min` | 40-50% | Todas |
| `7` / `maj7` / `min7` | 20-30% | 80%+ |
| `dim` / `aug` | 5-10% | 30%+ |
| `sus2` / `sus4` | 5-10% | 40%+ |
| `9th`, `11th`, `13th` | 5-10% | 20%+ |
| Inversões (bass) | 10-15% | 50%+ |

### Gêneros que Ajudam:

| Gênero | Acordes Comuns | Valor para Treino |
|--------|----------------|-------------------|
| **Jazz** | 7th, 9th, 11th, 13th, dim7 | 🔥🔥🔥 Excelente |
| **Bossa Nova** | maj7, min7, dim, aug | 🔥🔥🔥 Excelente |
| **MPB** | Variedade grande | 🔥🔥 Muito bom |
| **Pop/Rock** | maj, min, 7 | 🔥 Bom (base) |
| **Blues** | 7, 9 | 🔥🔥 Bom para dominantes |
| **Clássico** | dim, aug, inversões | 🔥🔥 Bom para raros |

---

## 📈 Estimativa Realista para Superar ChordFormer

### Com as Melhorias Propostas:

| Melhoria | Redução de Dados Necessários |
|----------|------------------------------|
| Reweighted Loss | -30% a -40% |
| Curriculum Learning | -10% a -20% |
| Data Augmentation (pitch) | -50% (equivale a 12x mais dados) |
| CRF Decoding | Não reduz, mas melhora qualidade |

### Cálculo Final:

```
ChordFormer: 1,217 músicas + pitch augmentation

BTC com melhorias:
- Base necessária: ~600-800 músicas
- Com pitch augmentation: ~300-400 músicas originais
- Com boa diversidade de gêneros: ~250-350 músicas
```

---

## 🎯 Recomendação Prática

### Mínimo para Resultados Competitivos:

| Requisito | Quantidade |
|-----------|------------|
| **Músicas totais** | 400-600 |
| **Com acordes 7th** | 80%+ (~400) |
| **Com acordes dim/aug** | 30%+ (~150) |
| **Com extensões (9th+)** | 20%+ (~100) |
| **Jazz/Bossa** | 15-20% (~80-100) |

### Distribuição Ideal do Dataset:

```
Total: 500 músicas

├── Pop/Rock: 200 (40%) - Base sólida de maj/min
├── Jazz: 100 (20%) - Acordes complexos
├── MPB/Bossa: 100 (20%) - Variedade brasileira
├── Blues: 50 (10%) - Dominantes e 9ths
└── Outros: 50 (10%) - Diversidade extra
```

---

## ⚖️ Trade-off: Quantidade vs Qualidade

### Opção A: Muitos Dados Simples

- 1,000+ músicas pop/rock
- Precisa de MUITO reweighted loss
- acc_class vai sofrer

### Opção B: Menos Dados Diversos (Recomendado)

- 400-600 músicas bem escolhidas
- Reweighted loss moderado
- Melhor equilíbrio acc_frame / acc_class

### Opção C: Híbrido

- 300 músicas diversas (core)
- 300 músicas pop (volume)
- + Data augmentation
- Melhor dos dois mundos

---

## 📝 Resumo Executivo

| Pergunta | Resposta |
|----------|----------|
| Quantas músicas mínimo? | **400-600** com boa diversidade |
| Quantas para superar ChordFormer? | **500-800** + melhorias implementadas |
| O que é mais importante? | **Diversidade > Quantidade** |
| Gêneros essenciais? | Jazz, Bossa Nova, MPB (acordes complexos) |
| Data augmentation ajuda? | **SIM**, equivale a 12x mais dados |

---

## 🔢 Fórmula Simplificada

```
Músicas Necessárias = 1200 / (1 + bonus_reweight + bonus_curriculum + bonus_augmentation + bonus_diversidade)

Onde:
- bonus_reweight = 0.4 (se implementar)
- bonus_curriculum = 0.2 (se ativar)
- bonus_augmentation = 1.0 (se usar pitch shift)
- bonus_diversidade = 0.3 (se tiver jazz/bossa)

Exemplo com tudo:
1200 / (1 + 0.4 + 0.2 + 1.0 + 0.3) = 1200 / 2.9 ≈ 414 músicas
```

---

## 📊 Tabela de Cenários

| Cenário | Músicas | Melhorias | Resultado Esperado |
|---------|---------|-----------|-------------------|
| Baseline (só dados) | 1,200+ | Nenhuma | = ChordFormer |
| Com reweight | 800 | Reweighted Loss | ≈ ChordFormer |
| Com reweight + curriculum | 600 | Reweight + Curriculum | ≈ ChordFormer |
| Completo | 400-500 | Todas as melhorias | > ChordFormer |
| Otimizado | 300-400 | Todas + diversidade | >> ChordFormer |

---

## 🎵 Checklist de Diversidade do Dataset

### Acordes Básicos (Obrigatório)
- [ ] maj em todas as 12 tonalidades
- [ ] min em todas as 12 tonalidades

### Acordes com Sétima (Muito Importante)
- [ ] 7 (dominante) - comum em blues/jazz
- [ ] maj7 - comum em bossa/jazz
- [ ] min7 - comum em jazz/MPB
- [ ] dim7 - comum em jazz
- [ ] hdim7 (meio-diminuto) - comum em jazz

### Acordes Suspensos (Importante)
- [ ] sus2 - comum em pop/rock
- [ ] sus4 - comum em pop/rock

### Acordes Alterados (Importante para Diversidade)
- [ ] dim - comum em passagens
- [ ] aug - comum em jazz

### Extensões (Diferencial)
- [ ] 9 - comum em jazz/funk
- [ ] maj9 - comum em jazz
- [ ] min9 - comum em jazz
- [ ] 11 - comum em jazz
- [ ] 13 - comum em jazz

### Inversões (Diferencial)
- [ ] Acordes com baixo diferente da fundamental
- [ ] Progressões com baixo cromático

---

## 💡 Conclusão

**Com as melhorias propostas (reweighted loss + curriculum + augmentation), você precisa de aproximadamente 400-600 músicas com boa diversidade de gêneros para ter resultados competitivos ou superiores ao ChordFormer.**

### Fórmula do Sucesso:

```
Resultado = Qualidade dos Dados × Técnicas de Balanceamento
          = (Diversidade + Gêneros Ricos) × (Reweight + Curriculum + Augmentation)
```

**O segredo não é ter MAIS músicas, é ter músicas CERTAS + técnicas que equilibram as predições!**

---

**Última atualização:** Dezembro 2024

