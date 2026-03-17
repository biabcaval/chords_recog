# Lógica de Saída Estruturada: Decomposição e Reconstrução de Acordes

Este documento explica em detalhe como o sistema converte entre representação textual de acordes e representação numérica usada pelo modelo, cobrindo os dois sentidos do fluxo.

---

## Visão Geral

```
TREINO (texto → números):
  "C:maj7(9)/E" → ChordDecomposer → {root:1, bass:4, triad:1, 7th:1, 9th:1, ...}

INFERÊNCIA (números → texto):
  {root:1, bass:4, triad:1, 7th:1, 9th:1, ...} → ChordReassembler → "C:maj79/E"
```

Arquivo de implementação: `utils/chord_decomposition.py`

---

## 1. O Vocabulário (CHORD_VOCAB)

Cada componente tem um vocabulário fixo onde a posição na lista é o índice numérico que o modelo usa:

| Componente | Classes | Índices |
|---|---|---|
| **root** | `N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B` | 0–12 |
| **bass** | `N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B` | 0–12 |
| **triad** | `N, maj, min, dim, aug, sus2, sus4` | 0–6 |
| **misc** | `N, 5` | 0–1 |
| **6th** | `N, 6` | 0–1 |
| **7th** | `N, 7, b7, bb7` | 0–3 |
| **9th** | `N, 9, #9, b9` | 0–3 |
| **11th** | `N, 11, #11` | 0–2 |
| **13th** | `N, 13, b13` | 0–2 |

**Total: 51 classes, 307.008 combinações possíveis.**

O índice `0` (`N`) em qualquer componente significa "ausente".

### Mapeamento reverso (CHORD_VOCAB_IDX)

Gerado automaticamente, inverte o dicionário para lookup rápido de label → índice:

```python
CHORD_VOCAB_IDX = {
    'root':  {'N': 0, 'C': 1, 'C#': 2, 'D': 3, ...},
    'triad': {'N': 0, 'maj': 1, 'min': 2, 'dim': 3, ...},
    '7th':   {'N': 0, '7': 1, 'b7': 2, 'bb7': 3},
    ...
}
```

---

## 2. Decomposição (ChordDecomposer)

Converte uma string de acorde em 9 índices numéricos. Usado no preprocessing e no dataset loader para gerar ground truth.

### 2.1 Parsing do label

O método `_parse_chord()` separa a string nos seus 3 elementos estruturais usando `:` e `/`:

| Input | root | quality | bass |
|---|---|---|---|
| `C:maj7` | `C` | `maj7` | — |
| `D:min7/F#` | `D` | `min7` | `F#` |
| `C/E` | `C` | — (assume maj) | `E` |
| `C` | `C` | — (assume maj) | — |
| `Bb:7(b9)` | `A#` (normalizado) | `7(b9)` | — |
| `N` | — | — | — |

Notas com bemol são normalizadas para sustenido: `Bb→A#`, `Eb→D#`, `Db→C#`, etc.

### 2.2 Decomposição da quality

O método `_decompose_quality()` analisa a string de qualidade e preenche triad + extensões. A **ordem de checagem é crítica** para evitar conflitos:

```
1. Power chord:  "5", "pedal"           → misc='5', retorna
2. Half-dim:     "hdim", "hdim7"        → triad='dim', 7th='b7'
3. Minor-major:  "minmaj7", "minmaj"    → triad='min', 7th='7'
4. Dim 7th:      "dim7"                 → triad='dim', 7th='bb7'
5. Major 7th:    "maj7"                 → triad='maj', 7th='7'
6. Tríade pura:  "sus2","sus4","maj","min","dim","aug"
7. Só extensão:  "7","9","13"           → assume triad='maj'
```

**Por que essa ordem?**

- `minmaj7` precisa ser checado antes de `min` e `maj` separados, senão `min` seria extraído primeiro e `maj7` seria interpretado errado.
- `maj7` precisa ser checado antes de `maj`, senão `maj` seria extraído e o `7` restante viraria `b7` (dominante) em vez de `7` (major 7th).
- `dim7` precisa ser checado antes de `dim`, para que a sétima diminuta (`bb7`) seja capturada junto.

### 2.3 Extração de extensões

O método `_extract_extensions()` processa o que sobrou após extrair a tríade. A **ordem de busca é do maior para o menor** para evitar matches parciais:

```
1. 13th:  b13, #13, 13    (extraído primeiro para não confundir "1" de "13" com outra coisa)
2. 11th:  #11, b11, 11    (idem: "11" contém "1")
3. 9th:   #9, b9, 9
4. 7th:   bb7, maj7, b7, 7  (só se ainda não foi setado nos passos especiais)
5. 6th:   6
```

Cada match é removido da string residual para não ser processado duas vezes.

### 2.4 Convenção da 7th

A nomenclatura da 7th pode confundir. A convenção usada:

| Nome no vocab | Significado musical | Intervalo | Quando aparece |
|---|---|---|---|
| `7` | Sétima **maior** | 11 semitons | `Cmaj7`, `Cminmaj7` |
| `b7` | Sétima **menor/dominante** | 10 semitons | `C7`, `Cmin7`, `Chdim7` |
| `bb7` | Sétima **diminuta** | 9 semitons | `Cdim7` |

Atenção: na notação popular, `C7` (sem "maj") implica sétima dominante (`b7`), não major. Por isso no parser, `'7'` sozinho vira `b7`.

### 2.5 Exemplos completos de decomposição

| Acorde | root | bass | triad | misc | 6th | 7th | 9th | 11th | 13th |
|---|---|---|---|---|---|---|---|---|---|
| `C:maj` | C(1) | N(0) | maj(1) | N(0) | N(0) | N(0) | N(0) | N(0) | N(0) |
| `G:min7` | G(8) | N(0) | min(2) | N(0) | N(0) | b7(2) | N(0) | N(0) | N(0) |
| `D:7` | D(3) | N(0) | maj(1) | N(0) | N(0) | b7(2) | N(0) | N(0) | N(0) |
| `A:min7/E` | A(10) | E(5) | min(2) | N(0) | N(0) | b7(2) | N(0) | N(0) | N(0) |
| `F:maj7` | F(6) | N(0) | maj(1) | N(0) | N(0) | 7(1) | N(0) | N(0) | N(0) |
| `C:maj7(9)` | C(1) | N(0) | maj(1) | N(0) | N(0) | 7(1) | 9(1) | N(0) | N(0) |
| `B:7(b9)` | B(12) | N(0) | maj(1) | N(0) | N(0) | b7(2) | b9(3) | N(0) | N(0) |
| `A:min6` | A(10) | N(0) | min(2) | N(0) | 6(1) | N(0) | N(0) | N(0) | N(0) |
| `E:5` | E(5) | N(0) | N(0) | 5(1) | N(0) | N(0) | N(0) | N(0) | N(0) |
| `C:13` | C(1) | N(0) | maj(1) | N(0) | N(0) | b7(2) | 9(1) | 11(1) | 13(1) |
| `Bb:hdim7` | A#(11) | N(0) | dim(3) | N(0) | N(0) | b7(2) | N(0) | N(0) | N(0) |
| `N` | N(0) | N(0) | N(0) | N(0) | N(0) | N(0) | N(0) | N(0) | N(0) |

---

## 3. Reconstrução (ChordReassembler)

Converte 9 índices numéricos (saída do modelo) de volta para uma string de acorde. Usado na inferência.

### 3.1 Regras de prioridade

A reconstrução segue regras musicais estritas, nesta ordem:

```
1. root = 'N'  →  retorna "N"                    (sem nota = sem acorde)
2. misc = '5'  →  retorna "{root}:5[/{bass}]"    (power chord tem prioridade)
3. triad = 'N' →  retorna "N"                    (sem tríade nem power = sem acorde)
4. Monta "{root}:{triad}"
5. Concatena extensões: 7th, 9th, 11th, 13th (se não forem 'N')
6. Adiciona "/{bass}" se bass ≠ root e bass ≠ 'N'
```

### 3.2 Por que essas regras?

- **Root obrigatório**: não existe acorde sem nota fundamental.
- **Power chord prioritário**: quando `misc='5'`, a tríade é irrelevante (power chord não tem terça).
- **Tríade obrigatória** (exceto power): um acorde precisa de pelo menos root + tríade para ter identidade harmônica.
- **Extensões opcionais**: 6th, 7th, 9th, 11th, 13th só aparecem se o modelo as prediz como presentes.
- **Bass condicional**: só aparece na notação se for diferente do root (evita `C:maj/C`).

### 3.3 Conversão shorthand → canônica

O decomposer adiciona **implied tones** para notações shorthand (não-parentéticas):

| Shorthand | Expansão canônica (decompose → reassemble) |
|---|---|
| `D:min9` | `D:min7(9)` — shorthand implica b7 |
| `C:maj9` | `C:maj7(9)` — shorthand implica 7 (major) |
| `C:9` | `C:7(9)` — shorthand implica b7 |
| `C:13` | `C:7(9)(11)(13)` — shorthand implica b7, 9, 11 |
| `C:maj(9)` | `C:maj(9)` — parentético = "add", sem implied tones |

### 3.4 Exemplos de reconstrução

| Componentes preditos | Acorde reconstruído |
|---|---|
| root=C, triad=maj, 7th=7 | `C:maj7` |
| root=G, triad=min, 7th=b7 | `G:min7` |
| root=D, triad=maj, 7th=b7 | `D:7` |
| root=A, triad=min, bass=E | `A:min/E` |
| root=E, misc=5 | `E:5` |
| root=C, triad=maj, 7th=7, 9th=9, 13th=13 | `C:maj7(9)(13)` |
| root=C, triad=maj, 7th=bb7 | `C:maj(bb7)` |
| root=A, triad=min, 6th=6, 7th=b7 | `A:min7(6)` |
| root=N | `N` |
| root=F, triad=N | `N` |

### 3.5 Confiança na reconstrução

O método `reassemble_with_confidence()` calcula um score de confiança para o acorde reconstruído. A regra é conservadora: a confiança final é o **mínimo** entre as confianças dos componentes ativos.

```
Componentes: root(conf=0.95), triad(conf=0.88), 7th(conf=0.72)
→ confiança do acorde = min(0.95, 0.88, 0.72) = 0.72
```

Isso garante que se qualquer componente está incerto, o acorde inteiro é marcado como incerto.

---

## 4. Processamento em Batch

### 4.1 decompose_batch()

Usado pelo dataset loader para processar todos os frames de uma janela de áudio:

```python
labels = ['C:maj7', 'C:maj7', 'G:min', 'N']
indices = decomposer.decompose_batch(labels)
# {'root': array([1, 1, 8, 0]),
#  'triad': array([1, 1, 2, 0]),
#  '7th': array([1, 1, 0, 0]),
#  ...}
```

### 4.2 reassemble_batch()

Usado na inferência para converter predições do modelo em acordes:

```python
predictions = {'root': array([1, 1, 8, 0]), 'triad': array([1, 1, 2, 0]), ...}
chords = reassembler.reassemble_batch(predictions)
# ['C:maj7', 'C:maj7', 'G:min', 'N']
```

### 4.3 reassemble_batch_2d()

Variante para batches com dimensão `(batch_size, seq_len)`. Retorna lista de listas.

---

## 5. Cobertura do Vocabulário

### 5.1 O que está coberto

O vocabulário cobre **100% das anotações** encontradas nos datasets padrão de chord recognition (Billboard, JAAH, RWC, Queen, Robbie Williams, Djavan), incluindo:

- Todas as tríades: maj, min, dim, aug, sus2, sus4
- Power chords (5)
- Sextas: maj6, min6
- Todas as sétimas: maj7, min7, dom7, dim7, hdim7, minmaj7
- Nonas: 9, #9, b9
- Décimas primeiras: 11, #11
- Décimas terceiras: 13, b13
- Inversões (slash chords) com qualquer nota no baixo
- Extensões parentéticas dos arquivos .lab: `(9)`, `(b9)`, `(#11)`, etc.

---

## 6. Relação com o Pipeline

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Arquivo .lab │────▶│  ChordDecomposer │────▶│  9 arrays de    │
│  "C:maj7(9)" │     │  decompose()     │     │  índices no .pt │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   DataLoader     │
                                              │  9 tensores      │
                                              │  (batch, seq)    │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   Modelo         │
                                              │  9 heads         │
                                              │  (softmax each)  │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐     ┌──────────────┐
                                              │ ChordReassembler│────▶│  "C:maj79"   │
                                              │ reassemble()    │     │  arquivo .lab │
                                              └─────────────────┘     └──────────────┘
```

---

## Referências

- Implementação: `utils/chord_decomposition.py`
- Modelo que consome os índices: `models/btc_model_decomposed.py`
- Dataset que gera os índices: `data/audio_dataset_structured.py`
- Inferência que reconstrói: `utils/decomposed_inference.py`
