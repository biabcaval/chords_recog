# Guia de Testes Rápidos - Quick Validation

## Overview

O módulo `quick_test_decomposed.py` permite validar toda a implementação **em menos de 2 minutos**, sem precisar de dados reais ou horas de treinamento.

---

## Como Usar

### Teste Básico (30 segundos)
```bash
python quick_test_decomposed.py
```

### Teste com Detalhes (1 minuto)
```bash
python quick_test_decomposed.py --verbose
```

### Teste em GPU Específica
```bash
python quick_test_decomposed.py --device cuda:0
```

---

## O Que é Testado

### 1. **Module Imports** (5 segundos)
Valida que todos os módulos podem ser importados:
- PyTorch
- NumPy
- Chord Decomposition
- Decomposed Model
- Structured Dataset
- Inference Utils

**Problema comum**: Dependências faltando
**Solução**: `pip install torch numpy librosa`

### 2. **Chord Decomposition** (10 segundos)
Testa decomposição de acordes e reassembly:
```python
C:maj  ->  {root: 'C', triad: 'maj', ...}  ->  C:maj
D:min7  ->  ...  ->  D:min7
E:maj9/G#  ->  ...  ->  E:maj9/G#
```

**Problema comum**: Parsing incorreto de acordes
**Solução**: Revisar formato (deve usar `:` separator)

### 3. **Model Architecture** (15 segundos)
Testa cada componente:
- ComponentHead (shape correto)
- MultiHeadChordDecomposer (8 heads)
- MultiTaskLoss (computável)

**Problema comum**: Shape mismatch nos tensores
**Solução**: Verificar feature_size e hidden_size na config

### 4. **Full Model Forward Pass** (20 segundos)
Testa modelo completo:
- Criação do modelo
- Forward pass sem labels
- Forward pass com labels
- Cálculo de loss
- Backward pass
- Optimizer step

**Problema comum**: Memória insuficiente
**Solução**: Reduzir batch_size ou hidden_size

### 5. **Inference Pipeline** (15 segundos)
Testa inferência:
- DecomposedChordInference
- Predição de acordes
- Decodificação
- Confidence scores
- ChordMetrics

**Problema comum**: Dimensões incompatíveis
**Solução**: Verificar que features têm formato (batch, 1, features, seq_len)

### 6. **Dataset Loading** (5 segundos)
Valida vocabulários:
- Root: 13 classes
- Bass: 13 classes
- Triad: 7 classes
- Misc: 2 classes
- Extensions: 4, 4, 3, 3 classes
- Total: 49 classes

**Problema comum**: Vocab sizes incorretos
**Solução**: Revisar CHORD_VOCAB em chord_decomposition.py

### 7. **Single Training Step** (15 segundos)
Simula um passo de treinamento real:
- Forward pass
- Loss computation
- Backward pass
- Gradient updates
- Verifica convergência

**Problema comum**: Loss infinito ou NaN
**Solução**: Verificar input data ou learning rate

---

## Tempo Estimado

| Teste | Tempo |
|-------|-------|
| Imports | ~5s |
| Decomposition | ~10s |
| Architecture | ~15s |
| Full Model | ~20s |
| Inference | ~15s |
| Dataset | ~5s |
| Training | ~15s |
| **Total** | **~90s (1.5 min)** |

---

## Exemplo de Output Bem-Sucedido

```
============================================================
QUICK VALIDATION TEST SUITE
============================================================

============================================================
TEST 1: Module Imports
============================================================
  ✓ PyTorch
  ✓ NumPy
  ✓ Chord Decomposition
  ✓ Decomposed Model
  ✓ Structured Dataset
  ✓ Inference Utils

============================================================
TEST 2: Chord Decomposition
============================================================
  ✓ C:maj  ->  C:maj
  ✓ D:min7  ->  D:min7
  ✓ E:maj9/G#  ->  E:maj9/G#
  ✓ F#:dim  ->  F#:dim
  ✓ N  ->  N
  ✓ All 5 test cases passed

============================================================
TEST 3: Model Architecture
============================================================
  Input shapes: batch=2, seq_len=50, features=192
  Testing ComponentHead...
    ✓ ComponentHead output: torch.Size([2, 50, 13])
  Testing MultiHeadChordDecomposer...
    ✓ MultiHeadChordDecomposer: 8 heads
    ✓ All 8 component heads present
  Testing MultiTaskLoss...
    ✓ MultiTaskLoss: 2.3456

============================================================
TEST 4: Full Model Forward Pass
============================================================
  Creating model with hidden_size=128
    ✓ Model created: 1,234,567 total params, 1,234,567 trainable
  Testing forward pass...
    ✓ Forward pass successful
    ✓ Predictions structure: Dict with 8 components
  Testing with labels...
    ✓ Loss computed: 2.1234
  Testing backward pass...
    ✓ Backward pass successful: 156 parameters with gradients
    ✓ Optimizer step successful

============================================================
TEST 5: Inference Pipeline
============================================================
  Testing DecomposedChordInference...
    ✓ Inference successful: 8 components
  Testing chord decoding...
    ✓ Chord decoding successful: 50 chords
      Example: C:maj7
  Testing probability computation...
    ✓ Confidence scores: min=0.112, max=0.889

============================================================
TEST 6: Dataset Loading
============================================================
  Testing component vocabulary sizes...
    ✓ root: 13
    ✓ bass: 13
    ✓ triad: 7
    ✓ misc: 2
    ✓ 7th: 4
    ✓ 9th: 4
    ✓ 11th: 3
    ✓ 13th: 3
  ✓ Total vocabulary size: 49 classes

============================================================
TEST 7: Single Training Step
============================================================
  Preparing dummy batch...
  Running training step...
    ✓ Forward pass: loss=2.3456
    ✓ Backward pass complete
    ✓ Optimizer step complete
    ✓ Second forward pass: loss=2.2234
    ✓ Loss changed (good): 2.3456  ->  2.2234

============================================================
TEST SUMMARY
============================================================
  ✓ Module Imports
  ✓ Chord Decomposition
  ✓ Model Architecture
  ✓ Full Model
  ✓ Inference Pipeline
  ✓ Dataset
  ✓ Training Step

Passed: 7/7

🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 
ALL TESTS PASSED! Ready for training.
🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 
```

---

## Exemplo de Output com Erro

```
============================================================
TEST 3: Model Architecture
============================================================
  Input shapes: batch=2, seq_len=50, features=192
  Testing ComponentHead...
✗ Error: RuntimeError: mat1 and mat2 shapes cannot be multiplied

============================================================
TEST SUMMARY
============================================================
  ✓ Module Imports
  ✓ Chord Decomposition
  ✗ Model Architecture
  ✗ Full Model
  ✗ Inference Pipeline
  ✓ Dataset
  ✗ Training Step

Passed: 2/7

⚠️  Some tests failed. Review errors above.
```

---

## Troubleshooting

### Erro: `ModuleNotFoundError: No module named 'torch'`
**Causa**: PyTorch não instalado
**Solução**:
```bash
pip install torch torchvision torchaudio
pip install numpy librosa
```

### Erro: `RuntimeError: CUDA out of memory`
**Causa**: Memória GPU insuficiente
**Solução**:
```bash
# Teste com CPU
python quick_test_decomposed.py --device cpu

# Ou reduza batch size (edite quick_test_decomposed.py)
batch_size = 1  # ao invés de 2
```

### Erro: `AssertionError: Wrong shape`
**Causa**: Dimensões de tensor incorretas
**Solução**:
```bash
# Execute com verbose para mais detalhes
python quick_test_decomposed.py --verbose

# Verifique as dimensões esperadas vs. obtidas
# Valide feature_size e hidden_size na config
```

### Erro: `KeyError: 'root'` ou similar
**Causa**: Componente faltando em CHORD_VOCAB
**Solução**:
```bash
# Verifique que todos os 8 componentes estão definidos
# em utils/chord_decomposition.py
```

### Teste lento (>5 minutos)
**Causa**: GPU compartilhada ou CPU lenta
**Solução**:
```bash
# Teste com batch menor
# Edite quick_test_decomposed.py linha ~200:
batch_size = 1
seq_len = 25  # reduzir de 50
hidden_size = 64  # reduzir de 128
```

---

## Quando Usar Este Teste

### Use ANTES de:
- Treinar por muito tempo
- Fazer commit de código
- Deployar em produção
- Compartilhar com outros

### Não é substituto para:
- Testes unitários (`test_decomposition.py`)
- Testes de integração (`infer_decomposed.py`)
- Avaliação em dataset real
- Testes de performance

---

## Fluxo de Desenvolvimento Recomendado

```
1. Fazer mudança no código
    ↓
2. Rodar: python quick_test_decomposed.py
    ↓
3. Passou?  ->  Treinar modelo
    ↓
4. Falhou?  ->  Debugar com --verbose
    ↓
5. Corrigir e voltar para step 2
```

---

## Exemplo Prático

### Cenário: Testar nova mudança rápido

```bash
# 1. Modificar código
# (editar utils/chord_decomposition.py, models/btc_model_decomposed.py, etc.)

# 2. Validar rápido
$ python quick_test_decomposed.py
Test 1: Imports ... ✓
Test 2: Decomposition ... ✓
Test 3: Architecture ... ✓
Test 4: Full Model ... ✓
Test 5: Inference ... ✓
Test 6: Dataset ... ✓
Test 7: Training ... ✓
👍 All tests passed!

# 3. Se passou, treinar
$ python train_decomposed.py --config run_config.yaml --num_epochs 100

# 4. Se falhou, debugar
$ python quick_test_decomposed.py --verbose
[Detalhes do erro]
```

---

## Comparação com Outros Testes

| Teste | Tempo | Cobertura | Dados |
|-------|-------|-----------|-------|
| quick_test_decomposed.py | ~1.5 min | 70% | Dummy |
| test_decomposition.py | ~30s | 40% | Dummy |
| train_decomposed.py (1 epoch) | ~30 min | 100% | Real |
| Full training | 4-24 horas | 100% | Real |

**Recomendação**: Use `quick_test_decomposed.py` antes de cada commit.

---

## Dicas

1. **Sempre rodar antes de fazer commits**
   ```bash
   python quick_test_decomposed.py && git commit
   ```

2. **Documentar erros encontrados**
   ```bash
   python quick_test_decomposed.py --verbose > test_log.txt
   ```

3. **Testar em múltiplos devices**
   ```bash
   python quick_test_decomposed.py --device cpu
   python quick_test_decomposed.py --device cuda:0
   ```

4. **Integrar com CI/CD**
   ```bash
   # .github/workflows/test.yml
   - name: Quick validation
     run: python quick_test_decomposed.py
   ```

---

## Output Files

Se for adicionar logging:
```bash
python quick_test_decomposed.py > test_results.log 2>&1
```

Arquivos que podem ser gerados:
- `test_results.log` - Log detalhado
- `test_errors.txt` - Apenas erros

---

## Conclusão

O módulo `quick_test_decomposed.py` é seu **melhor amigo** durante desenvolvimento:

- ✅ **Rápido** (~1.5 minutos)
- ✅ **Completo** (7 testes abrangentes)
- ✅ **Informativo** (detalhes com --verbose)
- ✅ **Fácil de usar** (apenas `python quick_test_decomposed.py`)

Use frequentemente! 🚀
