# NumPy Feature Roadmap

Priorisierte Liste fehlender NumPy-Funktionen für numpy-node.

## Bereits implementiert ✅

Diese Funktionen sind bereits verfügbar:

- **Array Creation:** `array`, `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `identity`, `empty`, `zerosLike`, `onesLike`, `emptyLike`
- **Arithmetic:** `add`, `subtract`, `multiply`, `divide`, `power` (+ in-place Varianten)
- **Unary Math:** `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `abs`, `negative`
- **Reductions:** `sum`, `prod`, `mean`, `std`, `variance`, `median`, `min`, `max`
- **Comparison:** `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`
- **Logical:** `logical_and`, `logical_or`, `logical_xor`, `logical_not`
- **Boolean Reductions:** `any`, `all`
- **Linear Algebra:** `matmul`, `dot`, `inv`, `det`, `solve`, `eig`, `eigvals`, `svd`, `qr`, `cholesky`, `norm`, `matrix_rank`, `trace`, `cond`, `lstsq`
- **Advanced:** `outer`, `kron`, `percentile`, `corrcoef`, `zscore`
- **Random:** `random.random`, `random.uniform`, `random.normal`, `random.randint`, `random.seed`

---

## Tier 1 - Essentiell (höchste Priorität)

Diese Funktionen werden in fast jedem NumPy-Projekt verwendet.

| Funktion                     | Beschreibung           | Anwendung                         | Komplexität |
| ---------------------------- | ---------------------- | --------------------------------- | ----------- |
| `argmin(a, axis?)`           | Index des Minimums     | Klassifikation, Nearest Neighbor  | Einfach     |
| `argmax(a, axis?)`           | Index des Maximums     | Predictions, Peak Detection       | Einfach     |
| `clip(a, min, max)`          | Werte begrenzen        | Gradient Clipping, Normalisierung | Einfach     |
| `where(cond, x, y)`          | Bedingte Auswahl       | Masking, Thresholding             | Mittel      |
| `concatenate(arrays, axis?)` | Arrays verbinden       | Batching, Sequenzen               | Mittel      |
| `stack(arrays, axis?)`       | Neue Achse + verbinden | Batch-Building                    | Mittel      |

## Tier 2 - Sehr häufig

Regelmäßig in Data Science und ML verwendet.

| Funktion             | Beschreibung        | Anwendung               | Komplexität |
| -------------------- | ------------------- | ----------------------- | ----------- |
| `cumsum(a, axis?)`   | Kumulative Summe    | Running Totals, CDF     | Einfach     |
| `cumprod(a, axis?)`  | Kumulatives Produkt | Wahrscheinlichkeiten    | Einfach     |
| `diff(a, n?, axis?)` | Differenzen         | Zeitreihen, Ableitungen | Einfach     |
| `sort(a, axis?)`     | Array sortieren     | Ranking, Median         | Mittel      |
| `argsort(a, axis?)`  | Sortier-Indizes     | Top-K, Ranking          | Mittel      |
| `unique(a)`          | Eindeutige Werte    | Kategorien, Labels      | Mittel      |

## Tier 3 - Häufig

Nützlich für viele Anwendungsfälle.

| Funktion                 | Beschreibung          | Anwendung                    | Komplexität |
| ------------------------ | --------------------- | ---------------------------- | ----------- |
| `round(a, decimals?)`    | Runden                | Darstellung, Diskretisierung | Einfach     |
| `floor(a)`               | Abrunden              | Integer-Konvertierung        | Einfach     |
| `ceil(a)`                | Aufrunden             | Integer-Konvertierung        | Einfach     |
| `searchsorted(a, v)`     | Einfügeposition       | Binning, Histogramme         | Mittel      |
| `tile(a, reps)`          | Array wiederholen     | Data Augmentation            | Mittel      |
| `repeat(a, reps, axis?)` | Elemente wiederholen  | Upsampling                   | Mittel      |
| `squeeze(a, axis?)`      | Einser-Dims entfernen | Shape Cleanup                | Einfach     |
| `expand_dims(a, axis)`   | Dimension hinzufügen  | Broadcasting Prep            | Einfach     |
| `vstack(arrays)`         | Vertikal stapeln      | Alias für concat axis=0      | Einfach     |
| `hstack(arrays)`         | Horizontal stapeln    | Alias für concat axis=1      | Einfach     |

## Tier 4 - Nützlich

Weniger häufig, aber wichtig für bestimmte Domains.

| Funktion                       | Beschreibung             | Anwendung             | Komplexität |
| ------------------------------ | ------------------------ | --------------------- | ----------- |
| `flip(a, axis?)`               | Array umkehren           | Bildverarbeitung      | Einfach     |
| `rot90(a, k?, axes?)`          | 90° rotieren             | Bildverarbeitung      | Mittel      |
| `sign(a)`                      | Vorzeichen               | Gradient Sign         | Einfach     |
| `mod(a, b)`                    | Modulo                   | Periodische Ops       | Einfach     |
| `allclose(a, b, rtol?, atol?)` | Approximative Gleichheit | Testing               | Einfach     |
| `isclose(a, b, rtol?, atol?)`  | Element-weise Nähe       | Numerische Vergleiche | Einfach     |
| `nonzero(a)`                   | Indizes != 0             | Sparse Ops            | Mittel      |
| `split(a, indices, axis?)`     | Array teilen             | Batch Splitting       | Mittel      |

## Tier 5 - Fortgeschritten

Komplex zu implementieren, aber eröffnet neue Domains.

| Funktion                        | Beschreibung           | Anwendung                    | Komplexität |
| ------------------------------- | ---------------------- | ---------------------------- | ----------- |
| `fft(a)`                        | Fast Fourier Transform | Audio, Signal Processing     | Hoch        |
| `ifft(a)`                       | Inverse FFT            | Frequenz → Zeit              | Hoch        |
| `rfft(a)`                       | Real FFT               | Effizienter für reelle Daten | Hoch        |
| `einsum(subscripts, *operands)` | Einstein Summation     | Flexible Tensor Ops          | Hoch        |

---

## Implementierungsplan

### Phase 1: Nächste Prioritäten (Tier 1)

1. `argmin`, `argmax` - C++ mit vDSP/BLAS Optimierung
2. `clip` - C++ mit vDSP_vclipD
3. `where` - Boolean Indexing (Boolean Arrays sind bereits implementiert ✅)
4. `concatenate`, `stack`

### Phase 2: Kumulative & Sortierung (Tier 2)

1. `cumsum`, `cumprod` - sequentiell, evtl. parallel prefix sum
2. `diff` - einfache Schleife
3. `sort`, `argsort` - std::sort oder vDSP_vsortD
4. `unique` - sort + deduplicate

### Phase 3: Utilities (Tier 3)

1. `round`, `floor`, `ceil` - vDSP oder std::round
2. `squeeze`, `expand_dims` - reine Shape-Manipulation
3. `tile`, `repeat` - Memory-Operationen
4. `searchsorted` - Binary Search

### Phase 4: Erweitert (Tier 4 & 5)

1. Array-Manipulation: `flip`, `rot90`, `split`
2. FFT: Accelerate Framework hat vDSP*fft*\*
3. `einsum`: Komplexer Parser + optimierte Kontraktionen

---

## Coverage Summary

| Kategorie          | Implementiert | Fehlend                             | Coverage |
| ------------------ | ------------- | ----------------------------------- | -------- |
| Array Creation     | 12            | 0                                   | 100%     |
| Arithmetic         | 5             | 0                                   | 100%     |
| Unary Math         | 8             | 3 (round, floor, ceil)              | 73%      |
| Reductions         | 8             | 4 (cumsum, cumprod, argmin, argmax) | 67%      |
| Comparison         | 6             | 2 (isclose, allclose)               | 75%      |
| Logical            | 4             | 0                                   | 100%     |
| Boolean Reductions | 2             | 0                                   | 100%     |
| Linear Algebra     | 15            | 0                                   | 100%     |
| Array Manipulation | 2             | 12+                                 | ~15%     |
| **Gesamt (Core)**  | **~62**       | **~25**                             | **~71%** |

_Hinweis: "Core" bezieht sich auf die am häufigsten verwendeten NumPy-Funktionen._
