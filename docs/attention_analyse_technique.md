# Analyse Technique et Formelle du Mécanisme d'Attention

> Document approfondi — Fondements mathématiques, justifications formelles et aspects techniques  
> Référence principale : Turner (2024), *An Introduction to Transformers*, arXiv:2304.10557v5

---

## Table des matières

1. [Définition formelle et notation](#1-définition-formelle-et-notation)
2. [Pourquoi le produit scalaire ?](#2-pourquoi-le-produit-scalaire)
3. [Le softmax : distribution de Boltzmann et smooth argmax](#3-le-softmax--distribution-de-boltzmann-et-smooth-argmax)
4. [Le scaling √d_k : preuve formelle](#4-le-scaling-√dk--preuve-formelle)
5. [Analyse des gradients](#5-analyse-des-gradients)
6. [Multi-Head Attention : justification par sous-espaces](#6-multi-head-attention--justification-par-sous-espaces)
7. [L'attention comme estimateur de Nadaraya-Watson](#7-lattention-comme-estimateur-de-nadaraya-watson)
8. [Connexion avec les mémoires associatives (Hopfield)](#8-connexion-avec-les-mémoires-associatives-hopfield)
9. [Propriétés algébriques : équivariance et invariance](#9-propriétés-algébriques--équivariance-et-invariance)
10. [Complexité computationnelle détaillée](#10-complexité-computationnelle-détaillée)
11. [Comparaison des variantes d'attention](#11-comparaison-des-variantes-dattention)
12. [Résultats d'approximation universelle](#12-résultats-dapproximation-universelle)

---

## 1. Définition formelle et notation

### 1.1 Cadre général

Soit une séquence d'entrée $X = (x_1, \ldots, x_n) \in \mathbb{R}^{n \times d}$, où $n$ est la longueur de la séquence et $d$ la dimension des représentations.

Le mécanisme d'attention est une fonction $\text{Attn} : \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$ définie par trois projections linéaires et une opération de mélange non-linéaire.

### 1.2 Projections Q, K, V

On définit trois matrices de projection apprenables :

$$W^Q \in \mathbb{R}^{d \times d_k}, \quad W^K \in \mathbb{R}^{d \times d_k}, \quad W^V \in \mathbb{R}^{d \times d_v}$$

qui produisent :

$$Q = XW^Q \in \mathbb{R}^{n \times d_k}, \quad K = XW^K \in \mathbb{R}^{n \times d_k}, \quad V = XW^V \in \mathbb{R}^{n \times d_v}$$

**Remarque formelle** : En self-attention, $Q$, $K$, $V$ sont dérivés de la même entrée $X$. En cross-attention, $Q = X_{\text{tgt}} W^Q$ tandis que $K = X_{\text{src}} W^K$, $V = X_{\text{src}} W^V$.

### 1.3 La formule d'attention

$$\text{Attention}(Q, K, V) = \underbrace{\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)}_{A \in \mathbb{R}^{n \times n}} V \in \mathbb{R}^{n \times d_v}$$

où $\text{softmax}$ est appliqué ligne par ligne :

$$A_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{l=1}^{n} \exp(q_i^\top k_l / \sqrt{d_k})}$$

La matrice $A$ est une **matrice stochastique à droite** : $A_{ij} \geq 0$ et $\sum_j A_{ij} = 1$ pour tout $i$.

### 1.4 Interprétation ligne par ligne

La sortie pour la position $i$ est :

$$\text{output}_i = \sum_{j=1}^{n} A_{ij} \, v_j = \sum_{j=1}^{n} \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{l} \exp(q_i^\top k_l / \sqrt{d_k})} \, v_j$$

C'est une **combinaison convexe** des values $v_j$, pondérée par la similarité entre la query $q_i$ et chaque key $k_j$.

---

## 2. Pourquoi le produit scalaire ?

### 2.1 Le produit scalaire comme mesure de similarité

Le score $s_{ij} = q_i^\top k_j = \|q_i\| \|k_j\| \cos\theta_{ij}$ mesure la **similarité directionnelle** entre query et key. Il satisfait les propriétés :

- **Symétrie** (sur les espaces non projetés) : utile pour la self-attention
- **Efficacité** : calculable par multiplication matricielle $QK^\top$ en $O(n^2 d_k)$
- **Différentiabilité** : gradient lisse par rapport à $Q$ et $K$

### 2.2 Alternatives considérées et rejetées

**Attention additive** (Bahdanau et al., 2015) :

$$s_{ij} = w^\top \tanh(W_1 q_i + W_2 k_j)$$

- Complexité : $O(n^2 d_k)$ mais avec des constantes plus élevées (pas de matmul optimisé)
- Plus expressive en théorie ($\tanh$ introduit une non-linéarité)
- En pratique, performances similaires au produit scalaire (Vaswani et al., 2017)

**Attention multiplicative générale** :

$$s_{ij} = q_i^\top W k_j, \quad W \in \mathbb{R}^{d_k \times d_k}$$

- Plus de paramètres ($d_k^2$ pour $W$)
- Le produit scalaire standard est le cas $W = I$

**Justification du choix** : le produit scalaire est le meilleur compromis expressivité/efficacité. Les GPU modernes sont optimisés pour les multiplications matricielles denses, ce qui rend $QK^\top$ extrêmement rapide.

### 2.3 Lien avec les noyaux (kernel methods)

Le score d'attention peut se voir comme un noyau :

$$\kappa(q_i, k_j) = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_l \exp(q_i^\top k_l / \sqrt{d_k})}$$

Ceci est un **noyau softmax normalisé**. Avant normalisation, $\exp(q^\top k / \sqrt{d_k})$ est un noyau exponentiel (lié au noyau RBF gaussien via l'identité $\exp(q^\top k) = \exp(-\|q-k\|^2/2) \cdot \exp(\|q\|^2/2) \cdot \exp(\|k\|^2/2)$).

La connexion est formellement établie par Tsai et al. (2019) : **l'attention est un kernel smoother dans l'espace des features**.

---

## 3. Le softmax : distribution de Boltzmann et smooth argmax

### 3.1 Définition et propriétés

Le softmax $\sigma : \mathbb{R}^n \to \Delta^{n-1}$ (le simplexe de probabilité) est défini par :

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$$

**Propriétés fondamentales** :
1. **Positivité** : $\sigma(z)_i > 0 \; \forall i$
2. **Normalisation** : $\sum_i \sigma(z)_i = 1$
3. **Monotonie** : $z_i > z_j \Rightarrow \sigma(z)_i > \sigma(z)_j$
4. **Invariance par translation** : $\sigma(z + c\mathbf{1}) = \sigma(z)$
5. **Différentiabilité** : $C^\infty$ partout

### 3.2 Le softmax comme distribution de Boltzmann

En physique statistique, la distribution de Boltzmann à température $T$ est :

$$p_i = \frac{e^{-E_i / T}}{\sum_j e^{-E_j / T}}$$

En identifiant $z_i = -E_i / T$, on voit que le softmax **est** la distribution de Boltzmann. Le facteur $1/\sqrt{d_k}$ dans l'attention joue le rôle d'une **température** : $T = \sqrt{d_k}$.

- $T \to 0$ ($\sqrt{d_k} \to 0$) : $\sigma(z/T) \to \text{one-hot argmax}$ (attention "dure")
- $T \to \infty$ ($\sqrt{d_k} \to \infty$) : $\sigma(z/T) \to \text{uniforme}$ (pas d'attention)
- $T$ modéré : distribution "douce" qui interpole entre les deux extrêmes

### 3.3 Le softmax comme approximation lisse du max

Le **log-sum-exp** (LSE) est une approximation lisse du max :

$$\text{LSE}(z_1, \ldots, z_n) = \log\!\left(\sum_{i=1}^n e^{z_i}\right) \approx \max(z_1, \ldots, z_n)$$

Plus précisément : $\max_i z_i \leq \text{LSE}(z) \leq \max_i z_i + \log n$.

Le softmax est le **gradient** du LSE :

$$\sigma(z) = \nabla_z \text{LSE}(z)$$

Cette propriété est importante car elle garantit que le softmax est le gradient d'une **fonction convexe**, ce qui a des implications pour la convergence de l'optimisation.

### 3.4 La Jacobienne du softmax

La matrice Jacobienne du softmax évalue comment de petits changements dans les logits affectent les probabilités de sortie :

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i (\delta_{ij} - \sigma_j) = \begin{cases} \sigma_i(1 - \sigma_i) & \text{si } i = j \\ -\sigma_i \sigma_j & \text{si } i \neq j \end{cases}$$

En notation matricielle : $J_\sigma = \text{diag}(\sigma) - \sigma \sigma^\top$.

**Conséquence critique** : si le softmax sature ($\sigma_i \approx 1$ pour un certain $i$), alors $\frac{\partial \sigma_i}{\partial z_j} \approx 0$ pour tout $j$. **Les gradients disparaissent**. C'est la raison fondamentale du scaling.

---

## 4. Le scaling √d_k : preuve formelle

### 4.1 Hypothèse et setup

**Hypothèse** : les composantes de $q$ et $k$ sont des variables aléatoires i.i.d. de moyenne 0 et variance 1 :

$$q = (q_1, \ldots, q_{d_k}), \quad k = (k_1, \ldots, k_{d_k}), \quad q_i, k_j \sim (0, 1) \text{ indépendants}$$

### 4.2 Calcul de la moyenne

$$\mathbb{E}[q^\top k] = \mathbb{E}\!\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} \underbrace{\mathbb{E}[q_i]}_{=0} \cdot \underbrace{\mathbb{E}[k_i]}_{=0} = 0$$

(par indépendance de $q_i$ et $k_i$)

### 4.3 Calcul de la variance

$$\text{Var}(q^\top k) = \text{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i)$$

(par indépendance des termes $q_i k_i$)

Pour un terme individuel :
$$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \cdot \mathbb{E}[k_i^2] - 0 = 1 \cdot 1 = 1$$

Donc :

$$\boxed{\text{Var}(q^\top k) = d_k}$$

### 4.4 Effet du scaling

En divisant par $\sqrt{d_k}$ :

$$\text{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \cdot \text{Var}(q^\top k) = \frac{d_k}{d_k} = 1$$

Le scaling **normalise la variance des scores à 1**, indépendamment de $d_k$.

### 4.5 Impact numérique concret

Pour $d_k = 64$ (notre configuration) :

| Métrique | Sans scaling | Avec scaling ($\div 8$) |
|----------|-------------|------------------------|
| $\text{Var}(s_{ij})$ | 64 | 1 |
| $\text{Std}(s_{ij})$ | 8 | 1 |
| Plage typique des scores | $[-24, +24]$ | $[-3, +3]$ |
| $\max(\sigma(s))$ typique | $\approx 0.999$ | $\approx 0.7$ |
| Norme du gradient du softmax | $\approx 10^{-10}$ | $\approx 0.2$ |

**Conclusion** : sans scaling avec $d_k = 64$, le softmax est **quasi-deterministe** : le gradient est de l'ordre de $10^{-10}$ — l'entraînement est impossible.

### 4.6 Remarque : pourquoi pas la normalisation L2 ?

Une alternative serait de normaliser : $s_{ij} = \frac{q_i^\top k_j}{\|q_i\| \|k_j\|}$ (cosine similarity).

Cette approche :
- Force les scores dans $[-1, 1]$, ce qui est stable
- Mais **supprime l'information de magnitude** : deux queries de normes très différentes produisent les mêmes poids d'attention
- Récemment revisitée dans QK-Norm (Henry et al., 2020) et dans certains LLMs modernes

Le scaling par $\sqrt{d_k}$ préserve les normes relatives tout en stabilisant la variance.

---

## 5. Analyse des gradients

### 5.1 Gradient de la loss par rapport aux scores

Soit $\mathcal{L}$ la loss. Le gradient par rapport à la matrice de scores $S = QK^\top / \sqrt{d_k}$ passe par le softmax.

Pour la ligne $i$ de la matrice d'attention :

$$\frac{\partial \mathcal{L}}{\partial S_i} = \frac{\partial \mathcal{L}}{\partial A_i} \cdot J_{\sigma}(S_i)$$

Avec $J_\sigma = \text{diag}(A_i) - A_i A_i^\top$ (cf. §3.4).

### 5.2 Gradient par rapport à Q et K

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}} \cdot \frac{\partial \mathcal{L}}{\partial S} \cdot K$$

$$\frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}} \cdot \left(\frac{\partial \mathcal{L}}{\partial S}\right)^\top \cdot Q$$

Le facteur $1/\sqrt{d_k}$ apparaît aussi dans le gradient (pas seulement dans le forward). C'est un scaling **symétrique**.

### 5.3 Gradient par rapport à V

$$\frac{\partial \mathcal{L}}{\partial V} = A^\top \cdot \frac{\partial \mathcal{L}}{\partial \text{output}}$$

Ici, $A^\top$ est une matrice stochastique à gauche. Le gradient de $V$ est une **combinaison convexe** des gradients de la sortie, ce qui a un effet régularisant naturel.

### 5.4 Le problème du gradient dans la matrice d'attention

Quand $A_{ij} \to 1$ (attention concentrée) :

$$J_\sigma \approx \text{diag}(e_j) - e_j e_j^\top$$

qui est une matrice de rang 1 avec des valeurs propres $\{0, 0, \ldots, 0, 1\}$ (en fait $\{0, \ldots, 1-\epsilon, \ldots, -\epsilon^2, \ldots\}$ pour $\epsilon$ petit). Les gradients sont **presque entièrement projetés sur un sous-espace de dimension 1**.

C'est pourquoi l'attention "hard" ($\text{argmax}$) n'est pas différentiable et nécessite des techniques comme le Straight-Through Estimator ou Gumbel-Softmax.

---

## 6. Multi-Head Attention : justification par sous-espaces

### 6.1 Formulation

$$\text{MultiHead}(Q, K, V) = [\text{head}_1; \ldots; \text{head}_h] \, W^O$$

$$\text{head}_i = \text{Attention}(X W_i^Q, \, X W_i^K, \, X W_i^V)$$

avec $W_i^Q, W_i^K \in \mathbb{R}^{d \times d_k}$, $W_i^V \in \mathbb{R}^{d \times d_v}$, $W^O \in \mathbb{R}^{hd_v \times d}$, et typiquement $d_k = d_v = d/h$.

### 6.2 Justification formelle : pourquoi $h$ têtes plutôt qu'une ?

**Théorème informel** (d'après Bhojanapalli et al., 2020) : Une attention single-head en dimension $d$ ne peut exprimer qu'une seule matrice d'attention $A \in \mathbb{R}^{n \times n}$ de rang au plus $d$. Avec $h$ têtes, on peut exprimer $h$ matrices d'attention indépendantes, chacune de rang au plus $d_k = d/h$, dont la combinaison via $W^O$ reconstruit une transformation de rang potentiellement $d$.

**Intuition géométrique** : chaque tête projette $Q$, $K$ dans un **sous-espace de dimension $d_k$**. L'attention est calculée dans cet espace réduit. Les $h$ têtes opèrent dans $h$ sous-espaces **différents** (appris), capturant des relations de nature différente.

### 6.3 Analyse du rang

Pour une seule tête avec $d_k = d$ :

$$\text{head} = \text{softmax}\!\left(\frac{(XW^Q)(XW^K)^\top}{\sqrt{d}}\right) XW^V$$

La matrice de poids $A = \text{softmax}(\ldots)$ est de rang au plus $n$ (trivial), mais les gradients sont dominés par la direction de plus forte attention.

Pour $h$ têtes avec $d_k = d/h$ :

$$\text{output} = \sum_{i=1}^{h} \text{head}_i \cdot W_i^O$$

Chaque $\text{head}_i \in \mathbb{R}^{n \times d_k}$ et $W_i^O \in \mathbb{R}^{d_k \times d}$. C'est une somme de $h$ termes de rang $\leq d_k$. Le rang total peut atteindre $\min(h \cdot d_k, d) = d$.

### 6.4 Le coût computationnel est identique

**Single-head** : $Q, K \in \mathbb{R}^{n \times d}$, donc $QK^\top$ coûte $O(n^2 d)$.

**Multi-head** : $h$ têtes, chacune avec $Q_i, K_i \in \mathbb{R}^{n \times d_k}$, coûtant $O(n^2 d_k)$ chacune. Total : $O(h \cdot n^2 d_k) = O(n^2 \cdot h d_k) = O(n^2 d)$.

$$\boxed{\text{Coût}(\text{multi-head}) = \text{Coût}(\text{single-head})}$$

On gagne en expressivité sans surcoût computationnel.

### 6.5 Analyse empirique : que capturent les têtes ?

Des travaux de Voita et al. (2019) et Clark et al. (2019) montrent que dans les modèles entraînés, les têtes se spécialisent :

- **Têtes positionnelles** : attention concentrée sur le voisinage local (tokens adjacents)
- **Têtes syntaxiques** : attention correspondant aux relations syntaxiques (sujet-verbe, etc.)
- **Têtes "rare tokens"** : forte attention sur les tokens rares ou informatifs
- **Têtes globales** : attention quasi-uniforme (contexte global)

**Pruning** : Voita et al. (2019) montrent que ~70% des têtes peuvent être supprimées après entraînement sans perte significative de performance, ce qui suggère une redondance, mais aussi que cette redondance facilite l'optimisation.

---

## 7. L'attention comme estimateur de Nadaraya-Watson

### 7.1 Rappel : l'estimateur de Nadaraya-Watson

En statistique non-paramétrique, l'estimateur de Nadaraya-Watson pour la régression $f(x) = \mathbb{E}[Y | X = x]$ est :

$$\hat{f}(x) = \frac{\sum_{i=1}^{n} K_h(x, x_i) \, y_i}{\sum_{i=1}^{n} K_h(x, x_i)}$$

où $K_h$ est un noyau (kernel) avec bande passante $h$.

### 7.2 L'attention **est** un estimateur de Nadaraya-Watson

En posant :
- $x \equiv q_i$ (query = point de requête)
- $x_j \equiv k_j$ (keys = points de données)
- $y_j \equiv v_j$ (values = réponses)
- $K_h(q, k) = \exp(q^\top k / \sqrt{d_k})$ (noyau exponentiel)

On obtient exactement :

$$\text{output}_i = \frac{\sum_{j=1}^{n} K(q_i, k_j) \, v_j}{\sum_{j=1}^{n} K(q_i, k_j)} = \text{Attention}(q_i, K, V)$$

**Implications** :
1. L'attention hérite des propriétés de consistance de l'estimateur NW
2. Le biais-variance trade-off est contrôlé par la "température" $\sqrt{d_k}$
3. Les projections $W^Q, W^K$ apprennent le noyau optimal pour la tâche

### 7.3 Connexion avec la théorie de la reproduction (RKHS)

Le noyau $\kappa(q, k) = \exp(q^\top k / \sqrt{d_k})$ est lié au noyau gaussien RBF via :

$$\exp\!\left(\frac{q^\top k}{\sqrt{d_k}}\right) = \exp\!\left(-\frac{\|q-k\|^2}{2\sqrt{d_k}}\right) \cdot \exp\!\left(\frac{\|q\|^2 + \|k\|^2}{2\sqrt{d_k}}\right)$$

Les termes $\exp(\|q\|^2/2\sqrt{d_k})$ et $\exp(\|k\|^2/2\sqrt{d_k})$ sont absorbés par la normalisation softmax. Donc l'attention softmax est **équivalente** à un kernel smoother gaussien après normalisation.

---

## 8. Connexion avec les mémoires associatives (Hopfield)

### 8.1 Réseau de Hopfield classique

Un réseau de Hopfield stocke $M$ patterns $\{\xi^\mu\}_{\mu=1}^M$ dans une matrice de poids :

$$W = \sum_{\mu=1}^{M} \xi^\mu (\xi^\mu)^\top$$

La règle de mise à jour pour retrouver un pattern à partir d'une entrée $x$ :

$$x^{(t+1)} = \text{sign}(W x^{(t)})$$

### 8.2 Le réseau de Hopfield moderne (Ramsauer et al., 2021)

Ramsauer et al. (2021) montrent qu'on peut définir un **réseau de Hopfield à continuité** dont la règle de mise à jour est :

$$x^{(t+1)} = \text{softmax}(\beta X^\top x^{(t)}) \cdot X$$

où $X = [\xi^1; \ldots; \xi^M]$ est la matrice des patterns stockés et $\beta$ est l'inverse de la température.

**C'est exactement la formule d'attention** avec :
- $x^{(t)} \equiv q$ (query = état courant)
- $X \equiv K$ (keys = patterns stockés)
- $X \equiv V$ (values = contenus à récupérer, si $K = V$)
- $\beta \equiv 1/\sqrt{d_k}$

### 8.3 Capacité de stockage

Le réseau de Hopfield classique stocke $O(d)$ patterns. Le réseau de Hopfield moderne (= attention softmax) peut stocker **exponentiellement** plus :

$$\text{Capacité} = O(e^{d/4})$$

C'est un résultat frappant : l'attention softmax est un mécanisme de mémoire associative avec une capacité exponentielle en la dimension.

### 8.4 Énergie et convergence

La règle de mise à jour de l'attention correspond à la minimisation d'une énergie :

$$E(q) = -\text{lse}(\beta K^\top q) + \frac{\beta}{2}\|q\|^2 + \text{const.}$$

où $\text{lse}$ est le log-sum-exp. Cette énergie est **convexe** en $q$ (car le lse est convexe et le terme quadratique aussi), ce qui garantit la convergence vers un minimum global unique.

---

## 9. Propriétés algébriques : équivariance et invariance

### 9.1 Équivariance par permutation

**Théorème** : L'attention (sans positional encoding) est **équivariante** par permutation.

*Preuve* : Soit $\Pi \in \{0,1\}^{n \times n}$ une matrice de permutation. Notons $f(X) = \text{Attn}(X)$.

$$f(\Pi X) = \text{softmax}\!\left(\frac{(\Pi X W^Q)(\Pi X W^K)^\top}{\sqrt{d_k}}\right) \Pi X W^V$$

$$= \text{softmax}\!\left(\frac{\Pi (X W^Q)(X W^K)^\top \Pi^\top}{\sqrt{d_k}}\right) \Pi X W^V$$

Comme le softmax est appliqué ligne par ligne et que $\Pi$ permute les lignes :

$$= \Pi \cdot \text{softmax}\!\left(\frac{(X W^Q)(X W^K)^\top}{\sqrt{d_k}}\right) X W^V = \Pi \cdot f(X) \quad \square$$

**Conséquence** : sans positional encoding, un Transformer ne distingue pas $[a, b, c]$ de $[c, a, b]$. Le positional encoding **brise** cette symétrie, ce qui est nécessaire pour les tâches séquentielles.

### 9.2 Ce que l'attention ne capture PAS seule

L'attention (une seule couche, sans FFN) ne peut calculer que des **fonctions linéaires des values** :

$$\text{output} = A V$$

La non-linéarité vient de $A$ (via softmax), mais $V$ subit une transformation **linéaire** (combinaison convexe). C'est pourquoi le **feed-forward network** après chaque couche d'attention est indispensable : il introduit une non-linéarité par position.

L'architecture complète — attention + FFN + résiduel — alterne :
1. **Mélange inter-positions** (attention) : les tokens se communiquent
2. **Transformation par position** (FFN) : chaque token est transformé non-linéairement

---

## 10. Complexité computationnelle détaillée

### 10.1 Décomposition par opération

Pour $n$ tokens, dimension $d$, et $d_k = d_v = d/h$ :

| Opération | Calcul | Complexité | Mémoire |
|-----------|--------|------------|---------|
| Projections $Q, K, V$ | $XW^Q$, $XW^K$, $XW^V$ | $O(nd^2)$ | $O(nd)$ |
| Scores $QK^\top$ | Multiplication matricielle | $O(n^2 d_k)$ par tête | $O(n^2 h) = O(n^2)$ |
| Softmax | Exponentielle + normalisation | $O(n^2)$ par tête | $O(n^2 h)$ |
| Pondération $AV$ | Multiplication matricielle | $O(n^2 d_v)$ par tête | $O(nd_v h) = O(nd)$ |
| Projection $W^O$ | Multiplication matricielle | $O(nd^2)$ | $O(nd)$ |
| **Total** | | $O(n^2 d + nd^2)$ | $O(n^2 + nd)$ |

### 10.2 Le goulot d'étranglement

- Si $n \ll d$ : le terme $O(nd^2)$ domine (projections)
- Si $n \gg d$ : le terme $O(n^2 d)$ domine (attention)
- Point de croisement : $n \approx d$

Pour notre toy_seq2seq : $n = 20$, $d = 64$, donc $n < d$ — les projections coûtent plus que l'attention elle-même. Pour les LLMs modernes avec $n = 8192+$, l'attention domine largement.

### 10.3 Mémoire : le vrai problème

Le stockage de la matrice $A \in \mathbb{R}^{n \times n}$ par tête est le **goulot d'étranglement mémoire** :

- $h = 4$ têtes, $n = 20$ patches : $4 \times 20^2 = 1600$ floats ≈ 6.4 KB → négligeable
- $h = 32$ têtes, $n = 4096$ tokens : $32 \times 4096^2 = 537M$ floats ≈ 2 GB → significatif

C'est ce problème qui motive **Flash Attention** (Dao et al., 2022), qui ne matérialise jamais $A$ en mémoire HBM grâce au tiling et au recomputing.

### 10.4 Parallélisme

**Parallélisme spatial** : les $n^2$ scores d'attention sont **indépendants** — calculables en $O(1)$ temps parallèle avec $O(n^2)$ processeurs. Contrairement aux RNN qui sont séquentiels ($O(n)$ temps même avec parallélisme infini).

**Parallélisme inter-têtes** : les $h$ têtes sont **parfaitement indépendantes** — pas de synchronisation nécessaire. C'est le reshape (batch, h, n, d_k) qui permet de les traiter comme un seul batched matmul.

---

## 11. Comparaison des variantes d'attention

### 11.1 Attention linéaire

Remplacement du softmax par une décomposition kernel :

$$\text{Attn}_{\text{lin}}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \phi(K)^\top \mathbf{1}}$$

où $\phi$ est une feature map (ex: $\phi(x) = \text{elu}(x) + 1$).

**Avantage** : complexité $O(nd^2)$ au lieu de $O(n^2 d)$ (on calcule $\phi(K)^\top V \in \mathbb{R}^{d \times d}$ d'abord).

**Inconvénient** : perte de la capacité de stockage exponentielle (Hopfield), performances inférieures sur les tâches nécessitant une attention "pointue".

### 11.2 Sparse Attention

$$A_{ij} = \begin{cases} \text{softmax}(\ldots) & \text{si } (i,j) \in \mathcal{S} \\ 0 & \text{sinon} \end{cases}$$

où $\mathcal{S}$ est un pattern de sparsité prédéfini (local, strided, random...).

- **Local** : chaque token attend ses $w$ voisins → $O(nw)$
- **Strided** : ajout d'attention à distance → $O(n\sqrt{n})$
- **Longformer** : local + global tokens → $O(n(w + g))$

### 11.3 Flash Attention (Dao et al., 2022)

Pas un changement mathématique mais une optimisation **algorithmique** :
- Même résultat que l'attention standard (exacte, pas approximée)
- Utilise le **tiling** : découpe $Q$, $K$, $V$ en blocs qui tiennent dans le SRAM du GPU
- Ne stocke jamais la matrice $A$ complète en HBM
- Recompute de $A$ pendant la backprop au lieu de la stocker

**Complexité mémoire** : $O(n)$ au lieu de $O(n^2)$ — **linéaire** !

**Complexité temporelle** : même $O(n^2 d)$, mais avec des constantes 2-4× meilleures grâce à moins d'accès HBM.

### 11.4 Tableau récapitulatif

| Variante | Temps | Mémoire | Exacte ? | Performance |
|----------|-------|---------|----------|-------------|
| Standard | $O(n^2 d)$ | $O(n^2)$ | ✓ | Référence |
| Flash Attention | $O(n^2 d)$ | $O(n)$ | ✓ | ≡ Standard |
| Linéaire | $O(n d^2)$ | $O(nd)$ | ✗ | ~95% |
| Sparse (local) | $O(nwd)$ | $O(nw)$ | ✗ | ~97% |
| Longformer | $O(n(w+g)d)$ | $O(n(w+g))$ | ✗ | ~98% |

---

## 12. Résultats d'approximation universelle

### 12.1 Les Transformers sont des approximateurs universels

**Théorème** (Yun et al., 2020) : Un Transformer avec suffisamment de couches et de têtes peut approximer uniformément toute fonction continue $f : [0,1]^{n \times d} \to \mathbb{R}^{n \times d}$ qui est **équivariante par permutation**, à précision $\epsilon > 0$ arbitraire.

Les ingrédients nécessaires sont :
1. Self-attention (pour le mélange inter-positions)
2. Feed-forward network (pour la transformation non-linéaire)
3. Connexions résiduelles (pour la profondeur)

### 12.2 Rôle spécifique de l'attention

L'attention seule (sans FFN) peut approximer universellement les fonctions de la forme :

$$g(X) = \sigma\!\left(\frac{XW^Q (XW^K)^\top}{\sqrt{d_k}}\right) X W^V$$

Ce sont des **moyennes pondérées contextuelles** — puissantes mais ne capturant pas toutes les transformations non-linéaires. C'est le FFN qui complète le pouvoir expressif.

### 12.3 Profondeur vs largeur

**Résultat** (Levine et al., 2020) : Un Transformer de profondeur $L$ peut exprimer des fonctions nécessitant $O(\exp(L))$ neurones dans un réseau peu profond. L'empilement de couches d'attention crée une **hiérarchie de dépendances** de portée croissante.

- Couche 1 : interactions directes entre paires de tokens
- Couche 2 : interactions entre "groupes" (via les représentations de la couche 1)
- Couche $L$ : dépendances d'ordre $L$ (interactions entre interactions entre...)

Pour notre modèle à 3 couches : suffisant pour que les patches-triangles détectent les autres patches-triangles et calculent leur moyenne.

---

## Synthèse

L'attention est bien plus qu'une heuristique :

| Perspective | Ce que l'attention **est** |
|------------|---------------------------|
| Algèbre linéaire | Multiplication de matrices stochastiques avec des values |
| Probabilités | Somme pondérée par une distribution softmax |
| Statistiques | Estimateur de Nadaraya-Watson avec noyau appris |
| Physique | Distribution de Boltzmann (mécanique statistique) |
| Mémoire | Réseau de Hopfield moderne (mémoire associative) |
| Optimisation | Gradient d'une fonction convexe (log-sum-exp) |
| Algèbre | Application équivariante par permutation |

Le scaling $\sqrt{d_k}$, souvent présenté comme une astuce, est en réalité une **nécessité statistique** (normalisation de la variance) et un **contrôle de température** (mécanique statistique) qui garantit l'entraînabilité du modèle.

Le multi-head, souvent vu comme un simple "truc pour plus de paramètres", est en réalité une **décomposition en sous-espaces** qui augmente l'expressivité sans surcoût, permettant de capturer simultanément des relations de natures différentes.

---

## Références

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Turner, R. E. (2024). *An Introduction to Transformers*. arXiv:2304.10557v5.
- Bahdanau, D. et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR.
- Ramsauer, H. et al. (2021). *Hopfield Networks is All You Need*. ICLR.
- Tsai, Y.-H. H. et al. (2019). *Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel*. EMNLP.
- Yun, C. et al. (2020). *Are Transformers Universal Approximators of Sequence-to-Sequence Functions?*. ICLR.
- Voita, E. et al. (2019). *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting*. ACL.
- Clark, K. et al. (2019). *What Does BERT Look At? An Analysis of BERT's Attention*. BlackboxNLP.
- Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.
- Levine, Y. et al. (2020). *Limits to Depth Efficiencies of Self-Attention*. NeurIPS.
- Bhojanapalli, S. et al. (2020). *Low-Rank Bottleneck in Multi-head Attention Models*. ICML.
