# Guide de Présentation : Attention & Multi-Head Attention
## Durée cible : ~20 minutes + 5 min questions

---

## Structure temporelle

| # | Section | Durée | Slides |
|---|---------|-------|--------|
| 1 | Rappel : du signal aux patches | 2 min | 1-2 |
| 2 | Intuition Q/K/V | 3 min | 3 |
| 3 | Formule + Code Scaled Dot-Product | 5 min | 4-6 |
| 4 | Multi-Head Attention : pourquoi et comment | 5 min | 7-10 |
| 5 | Le reshape expliqué | 2 min | 11 |
| 6 | Variantes (self, cross, masque) | 2 min | 12 |
| 7 | Application toy_seq2seq | 2 min | 13-14 |
| 8 | Synthèse + Complexité | 2 min | 15-16 |
| **Total** | | **~23 min** | |

---

## Partie 1 — Rappel : du signal aux patches (2 min)

### Ce qu'il faut dire :
> « La dernière fois, on a vu comment un signal brut de 100 points est découpé en 20 patches de 5 points chacun, puis projeté linéairement en dimension 64 avec un positional encoding. On se retrouve avec une matrice X de shape (20, 64). Mais à ce stade, chaque patch est traité indépendamment — il ne sait rien des autres. »

### Transition :
> « La question naturelle est donc : comment permettre à chaque patch de *consulter* les autres pour en extraire l'information dont il a besoin ? C'est exactement ce que fait le mécanisme d'attention. »

### Points clés à souligner :
- Montrer le diagramme signal → patches → matrice
- Insister sur le fait que sans attention, les patches sont isolés
- Donner les chiffres concrets : 100 points, 20 patches, dimension 64

---

## Partie 2 — L'intuition Q/K/V (3 min)

### Ce qu'il faut dire :
> « L'idée centrale est très simple. Imaginez une bibliothèque. Vous arrivez avec une *question* — c'est votre Query. Chaque livre a une *étiquette* — c'est sa Key. Et le livre lui-même, son contenu, c'est la Value. »

> « Vous comparez votre question à chaque étiquette pour obtenir un *score de pertinence*. Les livres les plus pertinents reçoivent un poids élevé. Ensuite vous faites une somme pondérée du contenu de tous les livres — vous obtenez une réponse qui combine les informations les plus utiles. »

### Exemple concret (Alammar) :
> « Prenez la phrase "The animal didn't cross the street because *it* was too tired". Quand le modèle traite le mot "it", l'attention lui permet de regarder tous les autres mots et de découvrir que "it" fait référence à "animal" et non à "street". C'est exactement ce mécanisme de Q/K/V qui permet ça. »

### Points clés :
- L'analogie de la bibliothèque est très parlante, s'appuyer dessus
- Montrer le diagramme sur la slide (Query → Keys → scores → Values → résultat)
- Faire le lien avec notre signal : un patch "triangle" interroge les autres patches pour trouver ceux qui lui ressemblent

---

## Partie 3 — Formule + Code Scaled Dot-Product (5 min)

### Étape 1 : La formule (1.5 min)
> « Mathématiquement, c'est élégant. On a la formule : Attention(Q, K, V) = softmax(QK^T / √d_k) × V. Décomposons : »

> « Étape 1 : on calcule le produit scalaire QK^T — ça nous donne un score pour chaque paire query-key. Étape 2 : on divise par racine de d_k — on verra pourquoi. Étape 3 : softmax pour normaliser en probabilités. Étape 4 : on multiplie par V pour obtenir la sortie. »

### Étape 2 : Le code (2 min)
> « Regardons le code de notre projet, fichier `attention.py`. Voici notre classe ScaledDotProductAttention. »

Parcourir le code ligne par ligne :
- `d_k = query.size(-1)` → « on récupère la dimension des keys »
- `scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)` → « c'est exactement QK^T/√d_k en un seul appel matmul »
- Le masque optionnel → « on remplit par -infini, le softmax les enverra vers 0 »
- `F.softmax(scores, dim=-1)` → « on normalise sur la dernière dimension »
- `torch.matmul(attention_weights, value)` → « et la somme pondérée des values »

### Étape 3 : Pourquoi √d_k ? (1.5 min)
> « C'est un détail crucial. Si d_k = 64, et que les composantes de q et k suivent une loi normale standard, alors le produit scalaire q·k a une variance de 64. Ça veut dire des scores autour de ±16. Le softmax de ±16, c'est quasiment du one-hot : un seul élément prend presque tout le poids. »

> « Et quand le softmax sature comme ça, les gradients sont quasi nuls — le modèle n'apprend plus rien. En divisant par √64 = 8, on ramène la variance à 1, le softmax reste "doux", et les gradients circulent correctement. »

### Points clés :
- **Astuce prof** : Si le professeur pose une question sur le scaling, c'est un point central qui montre une vraie compréhension
- Insister sur le lien formule ↔ code, c'est l'approche attendue
- Le masque sera re-détaillé dans la partie variantes

---

## Partie 4 — Multi-Head Attention (5 min)

### 4a : Motivation (1 min)
> « Avec une seule attention, le modèle fait *une seule* moyenne pondérée. Il ne peut capturer qu'un type de relation à la fois. Mais dans notre tâche, un patch pourrait avoir besoin de savoir : qui est mon voisin ? qui a la même forme ? qui a une hauteur similaire ? »

> « C'est pourquoi on utilise *plusieurs têtes* d'attention. Chaque tête apprend ses propres projections W_Q, W_K, W_V et se spécialise sur un aspect différent. »

### 4b : La formule (1 min)
> « La formule est simple : MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W^O. Chaque head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V). »

> « Le point crucial : avec 4 têtes et d_model = 64, chaque tête travaille en dimension d_k = 64/4 = 16. Le coût total est le même que si on faisait une seule attention en dimension 64 ! On ne paie pas plus cher. »

### 4c : Le pipeline visuel (1 min)
> « Regardons le pipeline complet sur le diagramme : X entre, on projette avec W_Q, W_K, W_V, on split en 4 têtes, chaque tête fait son attention indépendamment, on concatène les résultats, et on reprojette avec W_O pour revenir en dimension 64. »

### 4d : Le code __init__ (1 min)
> « Dans notre code, on voit les 4 projections linéaires W_q, W_k, W_v, W_o. L'astuce d'implémentation importante : on utilise un seul nn.Linear(64, 64) et pas 4 × nn.Linear(64, 16). Le split en têtes se fait par un simple reshape, pas par des matrices séparées. »

### 4e : Le code forward (1 min)
> « Le forward en 5 étapes : projection linéaire, reshape pour séparer les têtes, attention en parallèle grâce au batch, concat par reshape inverse, projection finale. »

> « Notez le `.view(batch_size, -1, num_heads, d_k).transpose(1, 2)` — c'est le cœur de l'implémentation efficace. »

---

## Partie 5 — L'astuce du reshape (2 min)

### Ce qu'il faut dire :
> « Ce reshape mérite qu'on s'y arrête parce que c'est souvent la partie qui crée de la confusion. »

> « Après la projection W_Q, on a un tenseur de shape (batch, 20, 64). Le .view le transforme en (batch, 20, 4, 16) — on découpe les 64 dimensions en 4 groupes de 16. Le .transpose permute les axes 1 et 2 pour obtenir (batch, 4, 20, 16). »

> « Pourquoi cette permutation ? Parce que torch.matmul fonctionne sur les deux dernières dimensions. Avec (batch, 4, 20, 16), le matmul fait 4 attentions en parallèle — une par tête — sans aucune boucle Python. C'est du batched matrix multiply. »

### Point clé :
- C'est une question fréquente et ça montre la compréhension de l'implémentation PyTorch
- Mentionner `.contiguous()` pour le concat inverse (mémoire non contiguë après transpose)

---

## Partie 6 — Variantes : Self, Cross, Masquage (2 min)

### Ce qu'il faut dire :
> « L'attention est utilisée trois fois dans un Transformer. Première fois : la self-attention dans l'encoder, où Q = K = V = X. Chaque position regarde toutes les autres — c'est ce qu'utilise notre toy_seq2seq. »

> « Deuxième : la masked self-attention dans le decoder. Même principe, mais la position i ne peut voir que les positions ≤ i. On met -infini dans la matrice des scores pour les positions futures, et le softmax les envoie à zéro. C'est le masque triangulaire qu'on voit sur le schéma. »

> « Troisième : la cross-attention. Le decoder fournit les queries, mais les keys et values viennent de l'encoder. C'est comme ça que le decoder "interroge" la représentation de l'entrée. »

---

## Partie 7 — Application toy_seq2seq (2 min)

### Ce qu'il faut dire :
> « Dans notre modèle, on utilise nn.TransformerEncoderLayer avec d_model=64, 4 têtes, 3 couches empilées. Chaque couche applique self-attention puis feedforward, avec des connexions résiduelles et LayerNorm. »

> « Concrètement, pour notre tâche de moyennage des pics par type : l'attention permet aux patches-triangles de se "trouver" entre eux, de partager leurs hauteurs, et de calculer la moyenne. De même pour les carrés. Le résultat : 98.2% de réduction de la loss après entraînement. L'attention est vraiment ce qui donne au modèle le pouvoir de résoudre cette tâche. »

---

## Partie 8 — Synthèse + Complexité (2 min)

### Récapitulatif rapide :
> « Pour résumer les 5 points clés : (1) l'attention c'est un mécanisme de recherche d'information via Q/K/V, (2) le scaling par √d_k est essentiel pour les gradients, (3) multi-head permet plusieurs perspectives pour le même coût, (4) trois variantes dans le Transformer, (5) l'implémentation repose sur le reshape et le batched matmul. »

### Complexité :
> « Un mot sur la complexité : l'attention est en O(n² × d), donc quadratique en la longueur de séquence. C'est pour ça que le patching est doublement utile : en plus de structurer l'entrée, il réduit n de 100 à 20, ce qui divise le coût de l'attention par 25. La prochaine fois, on pourra voir le feedforward et les connexions résiduelles qui complètent le bloc Transformer. »

---

## Questions potentielles du professeur + réponses

### Q: « Pourquoi le softmax et pas une autre normalisation ? »
> Le softmax garantit des poids positifs qui somment à 1 (distribution de probabilité). C'est interprétable et stable. D'autres normalisations (L1, L2) existent dans des variantes comme le linear attention, mais perdent certaines propriétés.

### Q: « Qu'est-ce qui se passe si on ne met pas le scaling ? »
> Le softmax sature : on obtient des vecteurs quasi one-hot. Les gradients sont presque nuls, le modèle n'apprend plus. C'est documenté dans le papier original de Vaswani et al. (2017) et repris par Turner §3.2.

### Q: « Pourquoi d_model doit être divisible par num_heads ? »
> Parce que d_k = d_model / num_heads doit être un entier. Si ce n'est pas le cas, le split en têtes est impossible avec un reshape simple. C'est validé par l'assert dans notre code.

### Q: « Quelle est la différence entre votre implémentation et celle de PyTorch ? »
> PyTorch utilise `nn.MultiheadAttention` qui fusionne W_Q, W_K, W_V en un seul `in_proj_weight` pour plus d'efficacité. Notre version sépare les 3 projections pour la lisibilité pédagogique. Le résultat mathématique est identique.

### Q: « Pourquoi 4 têtes et pas 8 comme dans le papier ? »
> Avec d_model=64, 4 têtes donnent d_k=16 ce qui est un bon compromis. 8 têtes donneraient d_k=8, ce qui est petit pour capturer des relations complexes. Le papier original utilise d_model=512 avec 8 têtes, donc d_k=64.

### Q: « Pouvez-vous expliquer le .contiguous() ? »
> Après un `.transpose()`, les données en mémoire ne sont plus contiguës (les strides changent). Le `.view()` qui suit a besoin d'un layout mémoire continu. `.contiguous()` copie les données dans un nouveau bloc mémoire avec le bon ordre.

### Q: « L'attention a-t-elle un biais inductif pour l'ordre ? »
> Non ! L'attention est invariante par permutation. C'est pourquoi le positional encoding (vu la séance précédente) est absolument nécessaire. Sans lui, le modèle ne distingue pas patch 1 de patch 20.

---

## Conseils de présentation

1. **Rythme** : Ne pas rusher les parties code. Le professeur préfère l'approche code, donc prendre le temps de commenter chaque ligne importante.

2. **Transitions** : Chaque partie doit s'enchaîner naturellement. Le fil rouge est : « les patches sont isolés → l'attention permet la communication → multi-head enrichit cette communication → voici comment c'est implémenté → voici ce que ça donne sur notre tâche ».

3. **Si vous êtes en avance** : Développer la partie reshape ou faire une petite démo live dans un notebook.

4. **Si vous êtes en retard** : Raccourcir la partie variantes (self/cross/mask) qui peut être mentionnée sans détailler.

5. **Interactivité** : Si le prof pose une question en cours de route, c'est bon signe. Répondre calmement en s'appuyant sur le code.

6. **Vocabulaire** : Utiliser les termes anglais (query, key, value, head, softmax) mais expliquer en français. Le prof appréciera la rigueur terminologique.
