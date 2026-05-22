# Report: TreeExplainer vs KernelExplainer per RandomSurvivalForest

**Progetto:** Foundation Model Clinical Data  
**File di riferimento:** `test_tabular_model.py`  
**Data:** 2026-05-11

---

## 1. Contesto

Il progetto confronta l'importanza delle feature cliniche originali (età, sesso, creatinina, ecc.) su diversi modelli di sopravvivenza:

- **DeepSurv** (semplice e vanilla) — addestrato su embeddings
- **Cox Proportional Hazard** — addestrato su embeddings
- **RandomSurvivalForest (RSF)** — addestrato su embeddings → `shap_fold_values["rsf"]`
- **RSF baseline** — addestrato su feature originali → `shap_fold_values["rsf_baseline"]`

Per tutti i modelli, il codice calcola i SHAP values tramite `KernelExplainer` e li aggrega con:

```python
shap_fold_values[key].append(np.abs(sv).mean(axis=0))
# risultato finale: vettore di shape (n_feature_originali,)
```

La domanda è: si può sostituire `KernelExplainer` con `TreeExplainer` per i casi RSF?

---

## 2. Differenze fondamentali tra i due Explainer

### 2.1 KernelExplainer

`KernelExplainer` è un explainer **model-agnostic**: riceve qualsiasi funzione `f(X) → output` come black box e approssima i valori di Shapley tramite campionamento e regressione lineare pesata.

- **Input:** una funzione Python arbitraria + un background dataset
- **Output:** SHAP values nel **medesimo spazio dell'input passato alla funzione**
- **Costo:** alto — richiede molte valutazioni della funzione (O(2^n) approssimate)
- **Garanzia:** produce veri valori di Shapley (assiomi: efficienza, simmetria, dummy, linearità)

### 2.2 TreeExplainer

`TreeExplainer` è un explainer **specifico per modelli ad albero**: sfrutta la struttura interna degli alberi per calcolare i valori di Shapley **esattamente** in tempo polinomiale (algoritmo TreeSHAP).

- **Input:** direttamente il modello ad albero + i dati nel suo spazio di input
- **Output:** SHAP values nel **medesimo spazio di input del modello ad albero**
- **Costo:** basso — algoritmo esatto O(TLD²) dove T=alberi, L=foglie, D=profondità
- **Garanzia:** valori di Shapley esatti (non approssimati)

---

## 3. Caso 1 — RSF con Embeddings: perché TreeExplainer NON è applicabile

### 3.1 La pipeline nel codice

```
Feature originali (X)
        │
        ▼
  _embed_fn(X)          ← rete neurale (TabPFN / TabICL / TabDPT)
        │
        ▼
  Embeddings            ← spazio latente (es. 512-dim per TabICL, ~192-dim per TabPFN)
        │
        ▼
  rsf.predict()         ← RSF addestrato SUGLI EMBEDDINGS
        │
        ▼
  Rischio scalare
```

Il codice attuale:

```python
# riga 839-842 di test_tabular_model.py
sv = shap.KernelExplainer(
    lambda X, _r=_rsf_emb_ref: _r.predict(_embed_fn(X)),  # pipeline completa
    background   # shape: (2, n_feature_originali)
).shap_values(X_shap_test)  # shape input: (2, n_feature_originali)
# sv shape: (2, n_feature_originali)
```

### 3.2 Perché TreeExplainer non può funzionare qui

`TreeExplainer` analizza la **struttura interna degli alberi** di RSF. Gli alberi di RSF sono stati costruiti sugli **embeddings**, quindi:

- Ogni split nei nodi è del tipo: `embedding_dim_47 < 0.23`
- Le feature su cui RSF ragiona sono le dimensioni latenti, non quelle cliniche

`TreeExplainer` non conosce e non può attraversare `_embed_fn`. Per usarlo bisognerebbe:

```python
# Approccio ipotetico — tecnicamente possibile ma ERRATO per l'obiettivo
embeddings_test = _embed_fn(X_shap_test)   # shape: (2, 512)
sv = shap.TreeExplainer(rsf).shap_values(embeddings_test)
# sv shape: (2, 512, n_time_points)  ← dimensioni latenti, non feature cliniche!
```

Il risultato spiegherebbe **quali dimensioni dell'embedding** contano per RSF, non quali feature cliniche originali. L'interpretazione sarebbe:

```
Importanza emb_0:  0.004
Importanza emb_1:  0.001
Importanza emb_47: 0.031   ← cosa rappresenta clinicamente emb_47? Non lo sappiamo.
...
```

### 3.3 L'impossibilità teorica di "risalire" alle feature originali

L'unico modo teorico per ottenere importanze nel spazio originale tramite TreeExplainer sarebbe applicare la **chain rule**:

```
∂output/∂feature_j = Σ_i ( ∂output/∂embedding_i  ×  ∂embedding_i/∂feature_j )
                            ─────────────────────     ─────────────────────────
                            TreeSHAP (fattibile)      Jacobiano di _embed_fn (problematico)
```

Questo approccio ha tre problemi fondamentali nel contesto del progetto:

**Problema 1 — Il Jacobiano di TabICL/TabPFN è ambiguo.**  
TabPFN e TabICL non sono reti neurali standard feed-forward. Utilizzano **attention sul training set come contesto**: per calcolare l'embedding di un campione di test, il modello guarda l'intero training set. Questo significa che `∂embedding_i/∂feature_j` dipende dal training set intero e non è localizzabile al singolo campione.

**Problema 2 — Il risultato non sarebbero veri valori di Shapley.**  
La chain rule produce un'attribuzione basata sui gradienti, non sui valori di Shapley. Violerebbero le proprietà di efficienza e simmetria che i SHAP values garantiscono.

**Problema 3 — Incompatibilità con gli altri modelli.**  
Tutti gli altri modelli (DeepSurv, Cox) usano `KernelExplainer` con output scalare sulle feature originali. Mischiare metodologie diverse invaliderebbe il confronto tra modelli.

### 3.4 Conclusione per il caso con embeddings

`KernelExplainer` con la pipeline completa (`_embed_fn` → `rsf.predict`) **è l'unica soluzione matematicamente corretta** per ottenere SHAP values nel spazio delle feature cliniche originali. Il costo computazionale più elevato è il prezzo necessario per la correttezza.

---

## 4. Caso 2 — RSF Baseline: perché TreeExplainer è applicabile ma produce risultati diversi

### 4.1 La pipeline nel codice baseline

```
Feature originali (X)
        │
        ▼  (nessuna trasformazione)
  rsf.predict()         ← RSF addestrato sulle FEATURE ORIGINALI
        │
        ▼
  Rischio scalare
```

Il codice attuale:

```python
# riga 863-866 di test_tabular_model.py
sv = shap.KernelExplainer(
    lambda X, _r=_rsf_base_ref: _r.predict(np.atleast_2d(np.asarray(X))),
    background
).shap_values(X_shap_test)
# sv shape: (2, n_feature_originali)
```

In questo caso, RSF riceve direttamente le feature cliniche originali. Gli split degli alberi sono del tipo:

- `creatinina < 1.5`
- `età < 65`
- `sesso == 0`

`TreeExplainer` potrebbe quindi in principio essere usato, e le feature sarebbero leggibili.

### 4.2 Il problema della struttura dell'output

RSF non è un regressore scalare standard: ogni foglia contiene una **funzione di sopravvivenza** (stimatore di Nelson-Aalen), non uno scalare. Quindi `TreeExplainer` restituisce un array **3D**:

```python
# Con KernelExplainer (attuale) — spiega rsf.predict() → scalare
sv = shap.KernelExplainer(
    lambda X, _r=rsf: _r.predict(np.atleast_2d(np.asarray(X))), background
).shap_values(X_shap_test)
# sv shape: (n_campioni, n_feature)         es. (2, 20)
# .mean(axis=0) → shape (20,)  ✓

# Con TreeExplainer — spiega le foglie di RSF → funzione di sopravvivenza
sv = shap.TreeExplainer(rsf).shap_values(X_shap_test)
# sv shape: (n_campioni, n_feature, n_time_points)  es. (2, 20, 87)
# .mean(axis=0) → shape (20, 87)  ✗  — incompatibile con il codice attuale
```

Per renderlo compatibile bisognerebbe aggregare anche sui time points:

```python
np.abs(sv).mean(axis=(0, 2))  # → shape (20,)  ✓ ma con semantica diversa
```

### 4.3 La differenza semantica nei risultati

Anche aggregando correttamente, i valori ottenuti hanno un significato diverso:

| | KernelExplainer | TreeExplainer |
|---|---|---|
| **Cosa spiega** | `rsf.predict(X)` — rischio scalare aggregato (media del rischio cumulativo su tutti i time points) | Funzione di sopravvivenza al tempo T — media su tutti i time points se si aggrega |
| **Domanda a cui risponde** | "Quale feature influenza di più il **rischio complessivo** del paziente?" | "Quale feature influenza di più la **sopravvivenza al tempo T**?" (mediata su T) |
| **Comparabilità con altri modelli** | Sì — tutti usano KernelExplainer su output scalare | No — semantica diversa dagli altri modelli |

### 4.4 Esempio numerico

Supponiamo un dataset con 3 feature (età, creatinina, sesso) e 87 time points:

```python
# KernelExplainer
sv = [[0.031, 0.087, 0.012],   # campione 1: (età, creatinina, sesso)
      [0.025, 0.091, 0.008]]   # campione 2
np.abs(sv).mean(axis=0)
# → [0.028, 0.089, 0.010]
#    età    creat  sesso   ← interpretabile, confrontabile con gli altri modelli

# TreeExplainer
sv = [[[0.030, 0.029, ..., 0.033],   # campione 1, età, 87 time points
       [0.085, 0.086, ..., 0.089],   # campione 1, creatinina, 87 time points
       [0.011, 0.012, ..., 0.013]],  # campione 1, sesso, 87 time points
      [...]]                          # campione 2
# shape: (2, 3, 87)

np.abs(sv).mean(axis=(0, 2))
# → [0.031, 0.087, 0.012]
#    età    creat  sesso   ← numericamente simile MA semanticamente diverso

# La domanda a cui risponde è diversa:
# KernelExplainer: "quanto conta la creatinina per il rischio aggregato?"
# TreeExplainer:   "quanto conta la creatinina per la sopravvivenza, mediata su 87 time points?"
```

### 4.5 Il vantaggio computazionale di TreeExplainer

Nel caso baseline TreeExplainer sarebbe comunque molto più veloce:

| | KernelExplainer | TreeExplainer |
|---|---|---|
| **Complessità** | O(2^n) approssimato — molte chiamate a `rsf.predict` | O(TLD²) esatto — analisi della struttura ad albero |
| **Tempo tipico** | Minuti per pochi campioni | Secondi per migliaia di campioni |
| **Approssimazione** | Sì (campionamento) | No (esatto) |

Tuttavia, il guadagno computazionale non giustifica la rottura della coerenza metodologica con gli altri modelli nel confronto.

---

## 5. Riepilogo

| Scenario | TreeExplainer applicabile? | Feature leggibili? | Risultato compatibile con il codice attuale? | Consigliato? |
|---|---|---|---|---|
| RSF con embeddings | **No** — RSF opera su dimensioni latenti | No | No | No |
| RSF baseline | **Sì** — RSF opera su feature originali | Sì | No (output 3D vs 2D) | No — rompe la coerenza metodologica |

### Raccomandazione

**Mantenere `KernelExplainer` per entrambi i casi RSF.** È l'unica scelta che garantisce:

1. SHAP values nel spazio delle feature cliniche originali (leggibili e interpretabili)
2. La stessa semantica (importanza per il rischio scalare) per tutti i modelli a confronto
3. Comparabilità diretta tra RSF, DeepSurv, Cox nel grafico finale delle importanze

Il costo computazionale più elevato di `KernelExplainer` è il prezzo corretto da pagare per la correttezza e la coerenza metodologica dell'analisi.

---

## 6. Riferimenti al codice

| Riga | Descrizione |
|---|---|
| `test_tabular_model.py:143` | Definizione di `feature_names_shap` (feature originali) |
| `test_tabular_model.py:787-788` | Definizione di `background` e `X_shap_test` nel spazio originale |
| `test_tabular_model.py:790-801` | Definizione di `_embed_fn` (TabPFN / TabICL / TabDPT) |
| `test_tabular_model.py:839-842` | SHAP RSF con embeddings (`KernelExplainer`) |
| `test_tabular_model.py:863-866` | SHAP RSF baseline (`KernelExplainer`) |
| `test_tabular_model.py:828` | Aggregazione: `np.abs(sv).mean(axis=0)` → shape `(n_features,)` |
| `src/tabicl/embedding.py:14-15` | Dimensione embedding TabICL: `embed_dim * row_num_cls` (tipicamente 128×4=512) |
