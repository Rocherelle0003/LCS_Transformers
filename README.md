# LCS_Transformers

Master 1 LCS (Lecture et Communication Scientifique). Ce projet accompagne l'étude approfondie de l'article "An Introduction to Transformers" de Richard E. Turner.

## 📚 Description

Ce repository contient des implémentations éducatives et des exemples avancés des concepts clés de l'architecture Transformer. Le code est conçu pour faciliter la compréhension profonde des mécanismes sous-jacents, avec des commentaires détaillés et des visualisations.

## 🏗️ Structure du Projet

```
LCS_Transformers/
├── src/
│   ├── transformers/           # Composants du Transformer
│   │   ├── attention.py        # Scaled Dot-Product & Multi-Head Attention
│   │   ├── positional_encoding.py  # Encodage positionnel
│   │   ├── feed_forward.py     # Réseau Feed-Forward
│   │   ├── encoder.py          # Encoder Transformer
│   │   ├── decoder.py          # Decoder Transformer
│   │   └── transformer.py      # Modèle complet
│   │
│   └── examples/               # Exemples éducatifs
│       ├── example1_attention.py     # Attention en profondeur
│       ├── example2_multihead.py     # Multi-Head Attention
│       ├── example3_architecture.py  # Architecture complète
│       ├── example4_translation.py   # Traduction seq2seq
│       ├── example5_variants.py      # Variantes modernes
│       ├── example6_illustrated_attention.py  # The Illustrated Transformer
│       ├── example7_annotated_transformer.py  # The Annotated Transformer
│       └── example8_advanced_concepts.py      # Concepts avancés (polygones, etc.)
│
├── tests/                      # Tests unitaires
│   └── test_transformers.py
│
├── requirements.txt            # Dépendances Python
└── README.md
```

## 🚀 Installation

```bash
# Cloner le repository
git clone https://github.com/Rocherelle0003/LCS_Transformers.git
cd LCS_Transformers

# Installer les dépendances
pip install -r requirements.txt
```

## 📖 Exemples Éducatifs

### Exemple 1: Attention Fondamentale
```bash
python -m src.examples.example1_attention
```
Explore le mécanisme d'attention scaled dot-product :
- Calcul des scores d'attention
- Importance du facteur d'échelle √d_k
- Masquage causal pour les modèles autoregressifs

### Exemple 2: Multi-Head Attention
```bash
python -m src.examples.example2_multihead
```
Comprendre l'attention multi-têtes :
- Pourquoi plusieurs têtes ?
- Spécialisation des têtes
- Visualisation des patterns d'attention

### Exemple 3: Architecture Complète
```bash
python -m src.examples.example3_architecture
```
L'architecture Transformer en détail :
- Structure Encoder-Decoder
- Connexions résiduelles et normalisation
- Encodage positionnel sinusoïdal

### Exemple 4: Traduction Seq2Seq
```bash
python -m src.examples.example4_translation
```
Exemple pratique de traduction nombre→mots :
- Préparation des données
- Boucle d'entraînement
- Décodage autoregressif

### Exemple 5: Variantes Modernes
```bash
python -m src.examples.example5_variants
```
Explorer les variantes du Transformer :
- Encoder-only (BERT-style)
- Decoder-only (GPT-style)
- Mécanismes d'attention efficaces
- Innovations architecturales modernes

### Exemple 6: The Illustrated Transformer 🆕
```bash
python -m src.examples.example6_illustrated_attention
```
Inspiré de [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) :
- Analogie Query-Key-Value (bibliothèque)
- Visualisation de l'attention sur des phrases réelles
- Patterns d'encodage positionnel
- Spécialisation des têtes d'attention

### Exemple 7: The Annotated Transformer 🆕
```bash
python -m src.examples.example7_annotated_transformer
```
Inspiré de [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) :
- Implémentation détaillée avec équations
- Label smoothing et ses effets
- Noam learning rate scheduler
- Classification de formes géométriques (12 types de polygones!)

### Exemple 8: Concepts Avancés (Fleuret) 🆕
```bash
python -m src.examples.example8_advanced_concepts
```
Inspiré des [slides de François Fleuret](https://fleuret.org/public/EN_20220809-Transformers/transformers-slides.pdf) :
- Attention comme dictionnaire soft
- Analyse de complexité O(n²)
- Biais inductifs des Transformers
- **Modélisation de séquences de polygones** :
  - Triangles (équilatéral, isocèle, scalène)
  - Quadrilatères (carré, rectangle, parallélogramme, losange, trapèze)
  - Polygones réguliers (pentagone, hexagone, heptagone, octogone)
  - Étoiles à 5 et 6 branches
  - Polygones irréguliers

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest tests/ -v

# Exécuter un test spécifique
pytest tests/test_transformers.py::TestScaledDotProductAttention -v
```

## 📊 Concepts Clés Implémentés

### 1. Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### 2. Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
où head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 3. Encodage Positionnel
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 4. Feed-Forward Network
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

## 📝 Références

- Turner, R. E. (2024). "An Introduction to Transformers"
- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Alammar, J. (2018). "[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)"
- Rush, A. (2018). "[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)"
- Fleuret, F. (2022). "[Transformers Slides](https://fleuret.org/public/EN_20220809-Transformers/transformers-slides.pdf)"

## 📄 Licence

GNU Affero General Public License v3.0
