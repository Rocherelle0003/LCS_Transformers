# LCS_Transformers

Master 1 LCS (Lecture et Communication Scientifique). Ce projet accompagne l'étude approfondie de l'article "An Introduction to Transformers" de Richard E. Turner.

##  Description

Ce repository contient des implémentations éducatives des concepts clés de l'architecture Transformer. Le code est conçu pour faciliter la compréhension profonde des mécanismes sous-jacents, avec des commentaires détaillés et des visualisations.

##  Structure du Projet

```
LCS_Transformers/
├── src/
│   ├── transformers/               # Composants du Transformer
│   │   ├── attention.py            # Scaled Dot-Product & Multi-Head Attention
│   │   ├── positional_encoding.py  # Encodage positionnel (sinusoïdal + appris)
│   │   ├── feed_forward.py         # Réseau Feed-Forward
│   │   ├── encoder.py              # Encoder Transformer
│   │   ├── decoder.py              # Decoder Transformer
│   │   └── transformer.py          # Modèle complet Encoder-Decoder
│   │
│   └── examples/                   # Exemples éducatifs
│       └── toy_seq2seq.py          # Tâche de transformation de signaux
│
├── docs/                           # Documentation pédagogique
│   ├── training_explained.py       # Explication ligne par ligne de l'entraînement
│   └── visualize_learning.py       # Visualisation de l'apprentissage
│
├── notebooks/                      # Notebooks Jupyter interactifs
│   └── toy_seq2seq.ipynb           # Exploration interactive du Toy Seq2Seq
│
├── tests/                          # Tests unitaires
│   └── test_toy_seq2seq.py
│
├── requirements.txt                # Dépendances Python
└── README.md
```

##  Installation

```bash
# Cloner le repository
git clone https://github.com/Rocherelle0003/LCS_Transformers.git
cd LCS_Transformers

# Installer les dépendances
pip install -r requirements.txt
```

##  Exemple : Toy Seq2Seq (Transformation de Signaux)

### Exécuter l'exemple principal
```bash
python -m src.examples.toy_seq2seq
```

**Tâche** : Transformer des signaux avec pics de hauteurs variées en signaux avec pics moyennés par type.
- **Input** : Signal avec pics triangulaires (▲) et carrés (■) de différentes hauteurs
- **Output** : Signal où tous les triangles ont la même hauteur (moyenne) et tous les carrés aussi

### Documentation pédagogique
```bash
# Explication détaillée de l'entraînement (ligne par ligne)
python -m docs.training_explained

# Visualisation de l'apprentissage
python -m docs.visualize_learning
```

### Notebook interactif
Ouvrir `notebooks/toy_seq2seq.ipynb` dans VS Code ou Jupyter.

##  Tests

```bash
# Exécuter tous les tests
pytest tests/ -v

# Exécuter un test spécifique
pytest tests/test_toy_seq2seq.py -v
```

##  Concepts Clés Implémentés

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
où head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Encodage Positionnel
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Feed-Forward Network
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

### Patching (dans Toy Seq2Seq)
Le signal est découpé en patches pour réduire la complexité :
- 100 points → 20 patches de 5 → Attention O(20²) au lieu de O(100²)

##  Références

- Turner, R. E. (2024). "An Introduction to Transformers"
- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Alammar, J. (2018). "The Illustrated Transformer"
- Rush, A. et al. (2018). "The Annotated Transformer"

##  Licence

Moi

