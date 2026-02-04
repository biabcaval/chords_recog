# Exemplos Práticos - Chord Structure Decomposition

## 1. Decomposição Básica de Acordes

```python
from utils.chord_decomposition import ChordDecomposer, ChordReassembler

# Inicializar decompositor
decomposer = ChordDecomposer()

# Decompor um acorde simples
chord = 'C:maj'
components = decomposer.decompose(chord)
print(components)
# Output: 
# {'root': 'C', 'bass': 'N', 'triad': 'maj', 'misc': 'N',
#  '7th': 'N', '9th': 'N', '11th': 'N', '13th': 'N'}

# Decompor um acorde complexo
chord = 'G:maj9/B'
components = decomposer.decompose(chord)
print(components)
# Output:
# {'root': 'G', 'bass': 'B', 'triad': 'maj', 'misc': 'N',
#  '7th': 'N', '9th': '9', '11th': 'N', '13th': 'N'}

# Decompor um lote de acordes
chord_list = ['C:maj7', 'D:min9', 'E:aug', 'F#:dim7/A', 'N']
components_batch = decomposer.decompose_batch(chord_list)
print(components_batch)
# Output (dicionário com arrays numpy):
# {
#     'root': array([0, 2, 4, 6, 0]),
#     'bass': array([0, 0, 0, 9, 0]),
#     'triad': array([1, 2, 4, 3, 0]),
#     ...
# }

# Remontar acorde a partir dos componentes
reassembler = ChordReassembler()
components = {'root': 'C', 'bass': 'N', 'triad': 'maj', 'misc': 'N',
              '7th': '7', '9th': 'N', '11th': 'N', '13th': 'N'}
chord = reassembler.reassemble(components)
print(chord)  # Output: 'C:maj7'
```

## 2. Carregamento de Dados com Decomposição

```python
from data.audio_dataset_structured import AudioDatasetStructured, AudioDataLoaderStructured
from utils.hparams import HParams

# Carregar configuração
config = HParams('run_config.yaml')

# Criar dataset de treinamento
train_dataset = AudioDatasetStructured(
    config,
    root_dir='/data/music/chord_recognition',
    dataset_names=('isophonic',),
    train=True,
    decompose=True,  # Ativar decomposição
    kfold=0
)

print(f"Número de amostras: {len(train_dataset)}")

# Acessar uma amostra
sample = train_dataset[0]
print(f"Features shape: {sample['feature'].shape}")
print(f"Components keys: {sample['components'].keys()}")
print(f"Root indices shape: {sample['components']['root'].shape}")

# Criar DataLoader com collate function personalizado
train_loader = AudioDataLoaderStructured(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterar sobre batches
for batch_idx, batch in enumerate(train_loader):
    features = batch['features']  # (batch, 1, feature_size, seq_len)
    components = batch['components']  # Dict[str -> (batch*seq_len,)]
    chord_lens = batch['chord_lens']
    boundaries = batch['boundaries']
    
    print(f"Batch {batch_idx}:")
    print(f"  Features: {features.shape}")
    print(f"  Root labels: {components['root'].shape}")
    print(f"  Triad labels: {components['triad'].shape}")
    
    if batch_idx >= 2:  # Mostrar apenas 3 batches
        break
```

## 3. Criar e Treinar o Modelo

```python
import torch
import torch.optim as optim
from models.btc_model_decomposed import BTC_model_decomposed, MultiTaskLoss
from utils.decomposed_inference import DecomposedChordTrainer

# Configuração do modelo
config = {
    'feature_size': 192,
    'hidden_size': 256,
    'num_layers': 4,
    'num_heads': 8,
    'total_key_depth': 256,
    'total_value_depth': 256,
    'filter_size': 1024,
    'timestep': 626,
    'input_dropout': 0.1,
    'layer_dropout': 0.1,
    'attention_dropout': 0.1,
    'relu_dropout': 0.1,
    'probs_out': False,
    'use_decomposition': True,
    'class_weight_gamma': 0.5,
    'class_weight_max': 10.0,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Computar pesos de classe
print("Computando pesos de classe...")
class_weights = MultiTaskLoss.compute_class_weights(
    train_dataset,
    gamma=config['class_weight_gamma'],
    w_max=config['class_weight_max'],
    device=device
)

print("Pesos de classe computados:")
for component, weights in class_weights.items():
    print(f"  {component}: min={weights.min():.3f}, max={weights.max():.3f}")

# Inicializar modelo
print("Inicializando modelo...")
model = BTC_model_decomposed(config, class_weights=class_weights)
model = model.to(device)

# Contar parâmetros
total_params = sum(p.numel() for p in model.parameters())
print(f"Total de parâmetros: {total_params:,}")

# Otimizador
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Trainer
trainer = DecomposedChordTrainer(model, device=device, verbose=True)

# Loop de treinamento
print("Iniciando treinamento...")
for epoch in range(5):  # 5 epochs para exemplo
    train_loss, _ = trainer.train_epoch(train_loader, optimizer, scheduler)
    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")
    
    # Validação a cada epoch
    val_metrics = trainer.validate(val_loader)
    print(f"  Val Loss = {val_metrics['val_loss']:.4f}")
```

## 4. Inferência em Arquivos de Áudio

```python
from infer_decomposed import ChordRecognitionInference
import torch

# Inicializar pipeline de inferência
inference = ChordRecognitionInference(
    config_path='run_config.yaml',
    checkpoint_path='checkpoints/model_best.pt',
    device='cuda'
)

# Reconhecer acordes em um arquivo
audio_file = '/data/music/songs/sample.mp3'
chord_sequence = inference.recognize_and_format(
    audio_file,
    aggregate=True  # Retornar apenas mudanças de acordes
)

# Exibir resultado
print("Sequência de acordes detectada:")
for start_time, end_time, chord, confidence in chord_sequence:
    duration = end_time - start_time
    print(f"{start_time:8.2f}s - {end_time:8.2f}s ({duration:5.2f}s): "
          f"{chord:15s} (confiança: {confidence:.3f})")

# Salvar resultado
with open('output.lab', 'w') as f:
    for start_time, end_time, chord, confidence in chord_sequence:
        f.write(f"{start_time:.2f} {end_time:.2f} {chord}\n")
```

## 5. Inferência em Batch

```python
from utils.decomposed_inference import DecomposedChordInference
import torch
import numpy as np

# Inicializar
inference = DecomposedChordInference(model, device='cuda')

# Preparar lote de features
batch_features = torch.randn(4, 1, 192, 626)  # 4 amostras
batch_features = batch_features.to(device)

# Opção 1: Obter previsões (índices)
predictions = inference.predict_batch(batch_features, return_probabilities=False)
print("Predictions (índices):")
for component, pred in predictions.items():
    print(f"  {component}: {pred.shape}")

# Opção 2: Obter probabilidades
probabilities = inference.predict_batch(batch_features, return_probabilities=True)
print("\nProbabilities:")
for component, probs in probabilities.items():
    print(f"  {component}: {probs.shape}, max={probs.max():.3f}")

# Opção 3: Decodificar para labels
chord_labels = inference.decode_predictions(predictions)
print(f"\nChord labels: {chord_labels}")

# Opção 4: Fazer tudo em um passo
chord_labels = inference.predict_and_decode(batch_features)
print(f"\nDirect prediction: {chord_labels}")

# Opção 5: Obter scores de confiança
confidences = inference.get_confidence_scores(probabilities)
print(f"\nConfidence scores: min={confidences.min():.3f}, "
      f"max={confidences.max():.3f}, mean={confidences.mean():.3f}")
```

## 6. Avaliação e Métricas

```python
from utils.decomposed_inference import ChordMetrics
import numpy as np

metrics_fn = ChordMetrics()

# Dados de exemplo
predictions = {
    'root': np.array([0, 2, 4, 0]),  # C, D, E, N
    'bass': np.array([0, 0, 0, 0]),
    'triad': np.array([1, 2, 4, 0]),  # maj, min, aug, N
    'misc': np.array([0, 0, 0, 0]),
    '7th': np.array([0, 3, 0, 0]),
    '9th': np.array([0, 0, 0, 0]),
    '11th': np.array([0, 0, 0, 0]),
    '13th': np.array([0, 0, 0, 0]),
}

targets = {
    'root': np.array([0, 2, 4, 0]),
    'bass': np.array([0, 0, 0, 0]),
    'triad': np.array([1, 2, 4, 0]),
    'misc': np.array([0, 0, 0, 0]),
    '7th': np.array([0, 0, 0, 0]),  # Diferente na segunda posição
    '9th': np.array([0, 0, 0, 0]),
    '11th': np.array([0, 0, 0, 0]),
    '13th': np.array([0, 0, 0, 0]),
}

# Computar métricas
metrics = metrics_fn.evaluate(predictions, targets)

print("Evaluation Metrics:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")

# Acurácia por componente
component_acc = metrics_fn.component_accuracy(predictions, targets)
print("\nComponent-wise Accuracy:")
for component, acc in component_acc.items():
    print(f"  {component}: {acc:.4f}")

# Acurácia geral (todos os componentes devem estar corretos)
chord_acc = metrics_fn.chord_accuracy(predictions, targets)
print(f"\nOverall Chord Accuracy: {chord_acc:.4f}")
```

## 7. Avaliação em Arquivo

```python
from infer_decomposed import ChordRecognitionInference

inference = ChordRecognitionInference(
    config_path='run_config.yaml',
    checkpoint_path='checkpoints/model_best.pt'
)

# Avaliar com referência
metrics = inference.evaluate_on_file(
    audio_path='/data/music/songs/sample.mp3',
    reference_path='/data/music/annotations/sample.lab'
)

print("Evaluation Results:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
```

## 8. Entendendo a Decomposição

```python
from utils.chord_decomposition import ChordDecomposer, COMPONENT_NAMES, CHORD_VOCAB

# Vocabulários disponíveis
print("Component Vocabularies:")
for component, vocab in CHORD_VOCAB.items():
    print(f"  {component} ({len(vocab)} classes): {vocab}")

# Entender mapeamento
decomposer = ChordDecomposer()

# Decomposição passo a passo
chord = 'G:maj9/D'
components = decomposer.decompose(chord)

print(f"\nDecompondo: {chord}")
for component in COMPONENT_NAMES:
    value = components[component]
    print(f"  {component:8s}: {value:8s} -> index {decomposer.vocab_idx[component][value]}")

# Remontar com índices
from utils.chord_decomposition import ChordReassembler
reassembler = ChordReassembler()

indices = {comp: decomposer.vocab_idx[comp][components[comp]] 
           for comp in COMPONENT_NAMES}
reassembled = reassembler.reassemble_from_indices(indices)
print(f"\nReassembled from indices: {reassembled}")
```

## 9. Tratamento de Prioridades

```python
from utils.chord_decomposition import ChordReassembler

reassembler = ChordReassembler()

# Exemplo 1: Se tríade é N, acorde é N
components = {
    'root': 'C',
    'bass': 'N',
    'triad': 'N',  # Nenhuma tríade = nenhum acorde
    'misc': '5',
    '7th': '7',
    '9th': 'N',
    '11th': 'N',
    '13th': 'N'
}
chord = reassembler.reassemble(components)
print(f"With triad=N: '{chord}'")  # Output: 'N'

# Exemplo 2: Bass diferente de root
components['triad'] = 'maj'
components['bass'] = 'E'
chord = reassembler.reassemble(components)
print(f"With bass=E: '{chord}'")  # Output: 'C:maj/E'

# Exemplo 3: Extensões em ordem
components['bass'] = 'N'
components['7th'] = 'b7'
components['9th'] = '9'
chord = reassembler.reassemble(components)
print(f"With extensions: '{chord}'")  # Output: 'C:majb79'
```

## 10. Entrenamento com Pesos Customizados

```python
from models.btc_model_decomposed import MultiTaskLoss
import torch

# Definir pesos customizados para componentes
component_weights = {
    'root': 1.5,    # Mais importante
    'bass': 1.0,
    'triad': 2.0,   # Muito importante (identidade do acorde)
    'misc': 0.5,    # Menos frequente
    '7th': 1.0,
    '9th': 1.0,
    '11th': 0.8,    # Extensões raras
    '13th': 0.8,
}

vocab_sizes = {
    'root': 13, 'bass': 13, 'triad': 7, 'misc': 2,
    '7th': 4, '9th': 4, '11th': 3, '13th': 3
}

# Criar loss com pesos customizados
loss_fn = MultiTaskLoss(
    vocab_sizes=vocab_sizes,
    gamma=0.5,
    w_max=10.0,
    component_weights=component_weights
)

# Testar com dados dummy
batch_size, seq_len = 4, 100

logits = {
    component: torch.randn(batch_size, seq_len, vocab_sizes[component])
    for component in vocab_sizes
}

labels = {
    component: torch.randint(0, vocab_sizes[component], (batch_size, seq_len))
    for component in vocab_sizes
}

# Computar loss
loss = loss_fn(logits, labels)
print(f"Loss com pesos customizados: {loss.item():.4f}")
```

---

## Resumo

Estes exemplos cobrem:
1. ✅ Decomposição básica de acordes
2. ✅ Carregamento de dados estruturados
3. ✅ Treinamento de modelo
4. ✅ Inferência em arquivos
5. ✅ Inferência em batch
6. ✅ Computação de métricas
7. ✅ Avaliação automática
8. ✅ Compreensão de vocabulários
9. ✅ Tratamento de prioridades
10. ✅ Treinamento customizado

Para mais detalhes, consulte `CHORD_DECOMPOSITION_GUIDE.md`.
