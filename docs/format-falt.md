# Faltwerk-Format `.falt` (früher `.csgs` / CurveStream-GS) – Bitstrom v1

Ein `.falt`-Abschnitt ist ein in sich beschriebener Byteblock für einen Abschnitt von `nframes ≤ L` Frames
(Paper §4.4). Ein Strom ist die Folge seiner Abschnitte. Code: `codec/falt.py` (Container), `codec/rans.py`
(Entropiecoder), `codec/prep.py` (numpy-Vorbereitung, nicht Teil des Decoders), `codec/measure.py` (Messung).
Alle Zahlen little-endian. Der Decoder rechnet nur mit Ganzzahlen < 2³² (Ausnahme: Zwischenprodukte < 2³¹ in
`rebuild`); er ist 1:1 nach TypeScript/WASM übertragbar (u32-Zustand, Byte-Ein-/Ausgabe, kleine Listen).

**Früher CSGS.** Bis 2026-11-02 hieß das Format CurveStream-GS: Dateiendung `.csgs`, Magic `CSGS`. Seit 2026-11-02
schreibt der Encoder Magic `FALT` in `s<index:04d>.falt`; alle Bytes nach der Magic sind unverändert. Ein
alter und ein neuer Abschnitt mit gleichem Inhalt unterscheiden sich daher nur in den 4 Magic-Bytes und in der CRC32
am Ende (sie deckt die Magic mit ab); `codec.falt.with_magic` wandelt in beide Richtungen um. Python-Decoder
(`codec/falt.py`) und WebGPU/TS-Player (`web/src/falt.ts`, Worker) lesen `FALT` und `CSGS`; ein Strom darf beide
mischen, Stromverzeichnisse dürfen `.falt` und `.csgs` enthalten (`codec.falt.section_files`, Manifest-Bau
`web/bench/make_viewer_data.py`), aber nicht beide Endungen für denselben Abschnitt. Vorhandene `.csgs`-Dateien werden
nicht umbenannt. Jede andere Magic wird vor der CRC-Prüfung mit
`not a Faltwerk section (.falt): magic b'…', expected b'FALT' or legacy b'CSGS'` abgewiesen.
Alte Namen bleiben als Aliase: Modul `codec.csgs` (= `codec.falt`), `web/src/csgs.ts`, `--write-csgs`, `P8_CSGS`,
Manifest-/Recipe-Panetyp `csgs`; Die Golden-Digests der Tests gelten jetzt für `FALT`;
die früheren `CSGS`-Digests prüfen die Tests weiter am umetikettierten Strom (`GOLDEN_CSGS`, `GOLDEN_V2_CSGS`,
`GOLDEN_V6_CSGS`).

## 1 Datenmodell eines Abschnitts

| Feld | Bedeutung |
|---|---|
| `ids` | globale Splat-IDs aller im Abschnitt lebenden Splats, streng steigend: erst die weiterlebenden (`n_cont`, geboren vor `g0`), dann die Neugeborenen mit fortlaufenden IDs `id_base, id_base+1, …` |
| `birth` | je Neugeborenem: Geburtsframe − `g0`, in `[0, nframes)` |
| `death` | je Splat: Todesframe − `g0` in `[1, nframes]` (stirbt im Abschnitt oder genau an seinem Ende) oder `nframes+1` = lebt in den nächsten Abschnitt weiter |
| `k[g]` | je Attributgruppe `g` und Splat: Stützpunktzahl K, eine der Zahlen aus `ks` |
| `knots[g]` | je Gruppe und Splat: K·dims Ganzzahlen, Stützpunkt-major (`k0d0 k0d1 … k1d0 …`), Wertebereich `[-2³⁰, 2³⁰)` |
| `sn` | je Splat: `n_sn` Werte 0…255 (statische höhere SH-Bänder, Koeffizient-major, 3 Farbkanäle je Koeffizient) |

Stützpunkte liegen gleichverteilt auf den Frames `g0 … g0+nframes−1` (Hutfunktionen wie `e6`/`e7`, `knots=gop`).
Dequantisierung je Skalar: `wert = offset + q · scale` (Tabelle im Kopf, Reihenfolge der Gruppen, danach SN);
Rotation (`R`, 4 Werte) wird nach dem Lerp normiert (nlerp). Der Container ist bezüglich dieser Ganzzahlen
verlustfrei; verlustbehaftet ist nur die Vorbereitung (Quantisierung, K-Wahl).

**Zustand** zwischen Abschnitten (Encoder und Decoder bauen ihn identisch aus dem dekodierten Abschnitt, `State.after`):
die Splats mit `death = nframes+1` in Abschnittsreihenfolge, je Splat ID, letzter Stützpunkt jeder Gruppe, SN und
K-Index der Gruppe 0. Genau diese Splats sind im Folgeabschnitt die `n_cont` weiterlebenden, in derselben Reihenfolge.

**Schlüsselabschnitt** (Flag `key`, Encoder: `index % key_interval == 0`): keine Referenz auf den Vorabschnitt.
Die IDs der weiterlebenden Splats werden gesendet, ihr erster Stützpunkt absolut (DPCM), ihr SN vollständig, der
K-Kontext ist „unbekannt“. Ein Decoder kann an jedem Schlüsselabschnitt einsteigen; danach ist sein Zustand gleich dem
eines Decoders, der von vorn dekodiert hat. Ein Nicht-Schlüsselabschnitt ohne seinen Vorgänger wird abgewiesen.

**Geteilter Randknoten** (Flag `shared_knot`): Der erste Stützpunkt eines weiterlebenden Splats ist der letzte des
Vorabschnitts und wird nicht übertragen (P1-Kurven). Ohne das Flag wird er als Differenz zu diesem letzten Stützpunkt
gesendet (e6-Symbole, Kurven nachträglich angepasst). Ruhende Splats (K=1, geteilter Knoten) kosten nur K-Wahl und
Todescode.

## 2 Byte-Layout

| Feld | Typ | Inhalt |
|---|---|---|
| magic | 4 B | `FALT` (vor geschrieben: `CSGS`, wird weiter gelesen) |
| version | u8 | 1 |
| flags | u8 | Bit 0 `key`, Bit 1 `shared_knot` |
| index, g0 | u32, u32 | Abschnittsnummer, erster Frame |
| L, nframes | u16, u16 | Nenn-Abschnittslänge (1…253), Frames in diesem Abschnitt (≤ L, letzter Abschnitt kürzer) |
| n_cont, n_new, id_base | 3 × u32 | |
| key_interval | u8 | N: Schlüsselabschnitt alle N Abschnitte |
| n_groups | u8 | dann je Gruppe: Kennbuchstabe u8 (ASCII, z. B. `P R S O D`), dims u8 |
| n_ks | u8 | dann `ks` als u8[n_ks] (zulässige K, z. B. 1 2 3 5 9) |
| n_sn | u16 | dann `sn_pred` u8[n_sn] (Median je SN-Skalar der gesendeten Splats) |
| offset, scale | f32[nq], f32[nq] | nq = Σ dims + n_sn |
| n_streams | u8 | = 5 + n_groups, dann Länge jedes Stroms u32[n_streams] |
| Ströme | | in dieser Reihenfolge: `birth`, `death`, `ids`, `k`, `knots_<g>` je Gruppe, `sn` |
| crc32 | u32 | zlib/PNG-CRC32 über alle Bytes davor |

Jeder Strom ist ein eigener rANS-Strom (leer = 0 Bytes) mit eigenen, pro Abschnitt frisch initialisierten Modellen;
die Ströme sind unabhängig dekodierbar (`knots_*` und `sn` brauchen nur `k`, `n_cont`, `n_new`), also auch parallel.

## 3 Entropiecoder (`codec/rans.py`)

- **rANS, byteweise** (Layout ryg_rans `rans_byte`): Zustand `x ∈ [2²³, 2³¹)`. Symbol mit (start, freq) bei
  Gesamtsumme 2^s (s = 15 für Modelle, s = n ≤ 15 für Rohbits mit freq = 1).
  Dekodieren: `slot = x & (2^s−1)`; Symbol suchen; `x = freq·(x >> s) + slot − start`; solange `x < 2²³`:
  `x = (x << 8) | nächstes Byte`. Stromanfang: 4 Bytes Zustand (little-endian). Ende: alle Bytes gelesen und
  `x == 2²³` (wird geprüft).
- **Adaptives Modell** (n Symbole): Zähler `c[i] = 1`, Summe `t = n`. Nach jedem Symbol `c[s] += 24`, `t += 24`; ist
  `t > 2¹⁶`, werden alle Zähler halbiert (`c = (c+1) >> 1`, t neu summiert). Die CDF wird bei der Erzeugung und dann
  nach 16, 32, 64, … 1024, 1024, … Symbolen neu gebaut: `f[i] = 1 + c[i]·(2¹⁵−n) // t`, der Rest bis 2¹⁵ geht an das
  häufigste Symbol (kleinster Index bei Gleichstand). Encoder und Decoder rufen dieselbe Aktualisierung auf.
- **Werte-Token** (Alphabet 49): vorzeichenbehaftete Werte zuerst Zickzack (`u = 2v` bzw. `−2v−1`); `u < 16` ist
  selbst das Token; sonst `w = u − 15`, `n = ⌊log2 w⌋`, Token `16 + n`, danach n Rohbits `w − 2ⁿ` (in Blöcken zu
  ≤ 15 Bit, höchstwertiger Block zuerst). Große Werte kosten so ein Präfix-Token plus Rohbits statt eines riesigen
  Alphabets.

## 4 Symbolreihenfolge und Kontexte

Splats immer in Abschnittsreihenfolge (`i = 0 … n−1`, weiterlebende zuerst).

| Strom | Symbole | Modell / Kontext |
|---|---|---|
| `birth` | je Neugeborenem `birth` (direkt, Alphabet nframes) | Kontext: `birth` des vorigen Neugeborenen (0 am Anfang) |
| `death` | je Splat der Todescode (direkt, Alphabet L+2) | weiterlebend / neugeboren |
| `ids` | nur Schlüsselabschnitt: je weiterlebendem Splat `id − vorige id` (erste gegen 0), Token | ein Modell |
| `k` | je Splat für jede Gruppe der Index von K in `ks` (direkt) | Gruppe 0: K-Index desselben Splats im Vorabschnitt (weiterlebend, kein Schlüssel), sonst „unbekannt“; Gruppe g>0: K-Index der Gruppe 0 desselben Splats |
| `knots_<g>` | je Splat: (a) erster Stützpunkt, (b) je Dimension c die Differenzen `knot[j][c] − knot[j−1][c]`, j = 1…K−1 | (a) weiterlebend ohne `shared_knot`: Differenz zum letzten Stützpunkt des Vorabschnitts, Modell [Residuum][c]; neugeboren oder Schlüssel: Differenz zum ersten Stützpunkt des zuletzt absolut gesendeten Splats derselben Gruppe (DPCM, Start 0), Modell [DPCM][c]; weiterlebend mit `shared_knot`: nichts. (b) Modell [K-Index][c][Bucket], Bucket 0 für j = 1, sonst `min(|vorige Differenz|, 2) + 1` |
| `sn` | Neugeborene (Schlüssel: alle): je Skalar `sn − sn_pred` | Modell [Skalar c][Bucket6 des Residuums von Skalar c−3] (gleicher Farbkanal, voriger Koeffizient; c < 3: Bucket 0); Bucket6: 0, 1, 2, 3–4, 5–8, ≥ 9 |

Alle Werte von `knots`, `ids` und `sn` laufen über das Werte-Token-Schema.

## 5 Messung

`python -m codec.measure --source tracks|curves …` vergleicht je Strom die rANS-Bytes mit zstd-19 über **dieselben
Ganzzahlen** (kleinste feste Breite, ein zstd-Rahmen je Strom und Abschnitt) und mit zstd-19 über alle Ströme eines
Abschnitts am Stück; `--source tracks` rechnet zusätzlich das Byte-Layout von `e6.encode_curves` nach (Gegenprobe der
e6-Zahl). Tabelle: `python codec/report.py <json>…`.

P2-Vorbereitung: `codec.measure --quat raw` sendet die unnormierten trainierten
Quaternionen × ein Skalar je Splat (der Decoder normiert nach dem Lerp ohnehin, die Rotation bleibt exakt die trainierte),
`--ktol n` wählt je Splat und Gruppe das kleinste K, dessen Kurve alle trainierten Stützpunkte auf n Schritte trifft
(Knoten 0 und letzter Knoten bleiben exakt), `shared_knot` folgt `soft_knot0` in `curves.npz` (weicher Randknoten → Knoten 0
als Residuum).

## 6 Grenzen v1

- Modelle starten in jedem Abschnitt neu (kein Übertrag adaptierter Modelle; kostet bei ~100K Splats wenig).
- Keine räumliche Vorhersage der ersten Stützpunkte von Neugeborenen außer DPCM in ID-Reihenfolge (→ §7: der Encoder
  darf die Neugeborenen räumlich nummerieren, ohne Formatwechsel).
- `L ≤ 253` (Geburts-/Todescodes direkt); Abschnittsgrößen bis 2³² Splats.
- Der Python-Decoder ist die Referenz, nicht schnell (~3–4 µs/Symbol). TypeScript-Port: `web/src/rans.ts`, `web/src/falt.ts`
  (bitgenau, `web/test/`).

## 7 Version 2: SH-Codebook (`SN_VQ`, Stufe S1)

Flag Bit 2 `SN_VQ` (Version 2 nur mit diesem Flag; Abschnitte ohne das Flag sind bytegleich v1, der Decoder liest beide).
Die höheren SH-Bänder eines Splats sind dann ein **SH-Grad** (0, 1, 2 oder 3 = die ersten 0 / 9 / 24 / 45 der `n_sn`
Skalare, Rest „Null“) plus je Grad-Stufe > 0 **ein Codebook-Index pro Stufe** (Residual-VQ). Die Codebooks stehen im
**Schlüsselabschnitt** und gelten für dessen Schlüsselintervall (Decoder-Zustand); Neugeborene in Delta-Abschnitten
werden gegen sie indiziert. Der Decoder liefert weiter `sn`-Zeilen (Rekonstruktion), daneben `sn_deg`, `sn_idx`, `sn_cb`.

Kopf-Erweiterung (nach `sn_pred`): `sn_zero` u8[n_sn] (Gitterpunkt des Werts 0,0 je Skalar, für abgeschnittene Bänder),
`n_levels` u8, je Stufe: `deg` u8, `n_stages` u8, `sizes` u16[n_stages] (Codebook-Größen ≤ 16384). Drei zusätzliche
Ströme nach `sn` (das in v2 leer bleibt): `sn_cb`, `sn_deg`, `sn_idx`; `n_streams` = 8 + n_groups.

| Strom | Symbole | Modell / Kontext |
|---|---|---|
| `sn_cb` | nur Schlüssel: je Stufe, Residualstufe, Eintrag, Skalar c: Stufe 0 `eintrag − sn_pred[c]` (0…255), Residualstufen der vorzeichenbehaftete Schritt (−255…255) | Stufe 0 dieselben Modelle wie `sn` [c][Bucket6 des Residuums von c−3]; Residualstufen eigene Modelle [c][Bucket6] |
| `sn_deg` | je kodiertem Splat (Neugeborene; Schlüssel: alle) die Stufennummer 0…n_levels (0 = Grad 0) | Kontext: Stufennummer des vorigen Splats |
| `sn_idx` | je kodiertem Splat mit Stufe > 0 je Residualstufe der Index (direkt, Alphabet = Größe) | ein adaptives Modell je (Stufe, Residualstufe) mit Zählerschritt 4 statt 24 (`rans.IDX_INC`: bei Alphabeten von Tausenden mittelt der Schätzer sonst über zu wenige Symbole, +0,6 Bit/Index bei 4096) |

`sn_pred` ist in v2 der Median der Stufe-0-Einträge je Skalar. Rekonstruktion je Splat (`csgs.vq_row`, nur Ganzzahlen):
Stufe 0 → `sn_zero`; sonst für die ersten `sn_dim(deg)` Skalare Summe der Codebook-Einträge über die Residualstufen,
geklemmt auf 0…255, dahinter `sn_zero`. Encoder-Seite (`codec/snvq.py`, numpy): Grad-Regel nach gewichteter Energie
der weggelassenen Bänder (Gewicht 1 oder Opazität × Fläche), gewichtetes k-means im Float-Raum je Stufe, Einträge aufs
SN-Gitter quantisiert, Zuordnung gegen die quantisierten Einträge; `--sn-vq-save/--sn-vq-from` (`codec.measure`)
sichern die Zuordnung samt SN-Gitter, damit ein Nachtraining mit festem SN (`E7_FINETUNE`, `E7_SN_FIXED`) denselben
SN-Teil des Bitstroms reproduziert. Bei Alphabeten > 64 sucht `AdaptiveModel.find` binär (gleiches Ergebnis).
TS-Decoder: `web/src/falt.ts` (bitgenau gegen `codec/falt.py`: synthetische
v2-Fixture `web/test/fixtures/synth_vq_*` mit Residual-VQ und frischen Codebooks je Schlüsselabschnitt).

## 8 Bewegungs-Prädiktion – Version 3, Flag `pred`

Ein Abschnitt mit Flag Bit 3 (`pred`, Wert 8) trägt `version = 3`; ohne das Flag ist der Abschnitt byteidentisch mit v1
(bzw. v2 mit SN-VQ). **Kombination:** `SN_VQ` (4) und `pred` (8) sind unabhängig und können zusammen gesetzt sein; die
Version ist die höchste, die ein gesetztes Flag verlangt (nur `SN_VQ` → 2, `pred` mit oder ohne `SN_VQ` → 3, kein Flag → 1);
der Decoder weist eine Version zurück, die nicht zu den Flags passt. Die Kopf-Erweiterungen stehen an ihren Plätzen
(`pred`-Block nach `ks`, SN-VQ-Block nach `sn_pred`) und ändern sich bei Kombination nicht; die Ströme beider Modi sind
disjunkt (`pred` ändert nur die `knots_*`-Ströme, SN-VQ nur `sn*`). Ein v1-Decoder weist Version 3 ab (klarer Fehler statt Fehldekodierung). Kopf: nach `ks` folgen
`knn_k` (u8) und je Gruppe ein Modus-Byte (Bitmaske):

| Bit | Name | Wirkung auf den Strom `knots_<g>` der Gruppe |
|---|---|---|
| 1 | `temporal` | Differenzen `d_j = knot[j]−knot[j−1]` werden ab j ≥ 2 als zweite Differenz `d_j − d_{j−1}` gesendet (Geschwindigkeits-Modell; j = 1 unverändert) |
| 2 | `spatial` | Vorhersage der Differenzen aus den `knn_k` nächsten **schon dekodierten** Splats (kausal in Abschnittsreihenfolge, Position = Stützpunkt 0 der Gruppe `P`, Abstand ganzzahlig, Gleichstand → kleinerer Index; `codec/knn.py`): deren Kurven werden auf das Knotenraster des Zielsplats umgetastet (`resample_knot`, lineare Interpolation, Rundung halb-auf), die Differenzen gemittelt (`predict_diffs`); gesendet wird `d_j − pred_j`, mit `temporal` zusätzlich `(d_j − d_{j−1}) − (pred_j − pred_{j−1})` für j ≥ 2 |
| 4 | `vel0` | Stützpunkt 0 eines weiterlebenden Splats (ohne `shared_knot`) wird gegen `letzter Stützpunkt + Endgeschwindigkeit` gesendet; Endgeschwindigkeit = letzte Knotendifferenz des Vorabschnitts auf einen Frame skaliert (`end_velocity`, Teil des Decoder-Zustands) |

Symbolreihenfolge im `pred`-Modus (alle Gruppen): zuerst Stützpunkt 0 **aller** Splats (wie v1: Residuum bzw. DPCM),
dann die Differenzen aller Splats – so kennt der Decoder vor der räumlichen Vorhersage alle Positionen. `knots_P` muss
vor jeder räumlich vorhergesagten Gruppe stehen; der Nachbarsatz wird einmal je Abschnitt aus `P` gebildet und für
alle Gruppen benutzt. Kontextmodelle wie v1 ([K-Index][Dimension][Bucket]), der Bucket bildet sich aus dem vorigen
**Residuum**. Die k-NN-Suche ist ein Gitter-Hash mit ganzzahliger Zellkante (Bbox-Regel, halbiert bis die belegten
Zellen im Mittel ≤ 4 Punkte tragen), Ringsuche mit exakter Abbruchregel und Brute-Force-Rückfall bei dünnen Gittern;
Encoder und Decoder rechnen dieselben ganzen Zahlen (Tests `tests/test_knn.py`, `tests/test_falt_pred.py`).

**Encoder-seitig ohne Formatwechsel (v1 bleibt):** `codec.prep … newborn_order='morton'` nummeriert die Neugeborenen
eines Abschnitts entlang einer Morton-Kurve ihrer quantisierten Position (Stützpunkt 0 von `P`); die IDs bleiben
fortlaufend, die weiterlebenden Splats behalten ihre Nummer. Die DPCM der ersten Stützpunkte läuft dann zwischen
räumlichen Nachbarn. `codec.measure --newborn-order morton --pred auto` wählt je Abschnitt und Gruppe `temporal` nur,
wenn der Strom dadurch kleiner wird (der Kopf trägt die Wahl).

TS-Decoder (`web/src/falt.ts`, `web/src/knn.ts`): alle Modi, jede Flag-Kombination. Bitgenau gegen `codec/falt.py`
(synthetische Fixtures `synth_pred*`/`synth_follow*` mit wechselnden Modi je Abschnitt, Gleichständen, Ausreißern und
`shared_knot`). Alles ist Ganzzahl-Arithmetik in
Doubles (`Math.floor` für `//`; exakt bis 2^53, daher Positionsspanne < 2^25 Schritte, sonst Fehler statt stiller Abweichung).
`causalKnn` liefert die exakte k-NN-Menge nach (d², Index) – unabhängig von Zellgröße und Suchreihenfolge (Test gegen
Brute-Force). Der Zustand trägt die Endgeschwindigkeiten immer mit (`pred auto` wechselt die Modi von Abschnitt zu Abschnitt).
## 9 Folge-Modus für verdeckte Splats – Version 4, Flag `FOLLOW`

v1-Abschnitte bleiben bytegleich (Golden-Test). Ein Abschnitt mit mindestens einem Folge-Splat trägt **Version 4** und
`FLAG_FOLLOW = 16`. **Umnummerierung bei der Integration:** Die Entwicklungsfassung hatte den Folge-Modus zunächst als Version 2 / Flag 4
geschrieben, beides belegt S1 (`SN_VQ`); es gibt keine ausgelieferten Folge-Daten, daher Flag 16 / Version 4. Die Flags
`SN_VQ` (4), `pred` (8) und `FOLLOW` (16) sind unabhängig und kombinierbar, die Version ist die höchste, die ein gesetztes
Flag verlangt (2 / 3 / 4); der Decoder weist unpassende Versions-/Flag-Paare ab. Ein Decoder, der Version 4 nicht kennt,
lehnt den Abschnitt über das Versionsbyte ab.

- Datenmodell: `Section.follow` = je Splat eine Bitmaske über die Stützpunkte der Gruppe `P` (Bit j = Knoten j).
  Knoten 0 ist nie maskiert, Masken nur bei K ≥ 2. Für maskierte Knoten sendet der Strom **keine** Positionsdaten.
- Strom `follow` (immer der **letzte** Strom, nach `sn` bzw. `sn_idx`): für jeden Splat mit K_P ≥ 2 die Bits j = 1..K_P−1, Kontext
  (K-Index der Gruppe P, Bit des vorigen Knotens), binäres adaptives Modell.
- Knotenstrom `knots_P`: maskierte Knoten werden als ihr Anker gesendet (Anker = nächster unmaskierter Knoten darunter),
  also mit Differenz 0; der Decoder ersetzt sie danach durch die Vorhersage.
- Vorhersage `apply_follow` (exakte Ganzzahlen, im Encoder und Decoder identisch): Nachbarn = die `FOLLOW_K = 4` Splats
  ohne Maske, deren Knoten-0-Position (quantisiert) dem Knoten 0 des Folge-Splats am nächsten liegt (quadratischer
  Abstand, Gleichstand → kleinerer Splat-Index; Suche über Gitterzellen von `FOLLOW_CELL = 8` Quantisierungsschritten (nur Laufzeit, das Ergebnis ist exakt),
  bei dünner Besetzung Vollsuche – beides exakt). Gemeinsames Knotenraster D = max(ks)−1 (ks 1,2,3,5,9 → 8), Knoten j
  eines Splats mit K Knoten liegt bei u = j·D/(K−1); V(u) = D × lineare Interpolation der Nachbarkurve bei u.
  Für maskierten Knoten j mit Anker a und je Achse: `x_j = x_a + floor((S + D·n/2) / (D·n))`,
  `S = Σ_Nachbarn (V(u_j) − V(u_a))`, n = Zahl der Nachbarn (0 → der Splat hält x_a). Folge-Splats sind nie Nachbarn,
  die Reihenfolge spielt keine Rolle. `floor` ist mathematisches Abrunden (auch für negative Zahlen; im TS-Port
  `Math.floor` auf BigInt/Number achten).
- Encoder-Vertrag: `Section.knots` enthält bereits die Vorhersage (`prep` ruft `apply_follow` auf); `_check` lehnt
  andere Werte ab. Damit sind `State.after` (letzter Knoten für den nächsten Abschnitt) auf beiden Seiten gleich.
- Erzeugung: `p8_e7.py` mit `E7_DORMANT=1` schreibt je Abschnitt `g<i>_dk` (ruhende Knoten) in `curves.npz`;
  `codec/prep.sections_from_curves(follow=True)` macht daraus die Masken (Folge-Splats behalten für P das trainierte K),
  `codec.measure --no-follow` sendet dieselben Knoten explizit (ohne Flag) zum Ratenvergleich.
- Web-Decoder (`web/src/falt.ts`): portiert – Strom `follow` (letzter Strom) lesen, `knots_P` wie v1/`pred` dekodieren, danach
  `applyFollow` (exakte Ganzzahlen in Doubles, `Math.floor`; die Nachbarmenge ist die exakte k-NN, die Gitterzelle beeinflusst nur
  die Laufzeit). Bitgenau gegen Python (synthetische Fixtures `synth_follow*`, v4 mit/ohne `pred`/`SN_VQ`/`shared_knot`; Masken auf
  höchstens 32 Stützpunkte je Splat begrenzt).
- Nicht in Version 4: allgemeine Bewegungsvorhersage aus Nachbarn/Bewegungsbasen für sichtbare
  Splats (das macht §8 `pred` teilweise), Langzeit-Referenz/Wiederbelebung gestorbener Splats per ID über Abschnitte.
- Zusammenspiel mit `pred`: die Knotenströme tragen für Folge-Splats die gehaltenen Ankerwerte (wie oben), `pred` kodiert
  diese; der Decoder wendet `apply_follow` nach dem Dekodieren an, `vel0`/Zustand sehen die vorhergesagten Knoten.

## 10 Quantisierungstabelle je Abschnitt – Version 5, Flag `RESCALE`

Der Kopf trug `offset`/`scale` (§2) schon in **jedem** Abschnitt; ein Encoder darf sie je Abschnitt ändern (Positionsschritt
`q_pos`, Gitter von S/O/D, Rotationseinheit, K-Toleranz = nur Encoder). Der Decoder dequantisiert mit der Tabelle des eigenen
Abschnitts. Was dabei fehlte: die Ganzzahlen des Vorgängers (`State.last`, Endgeschwindigkeit) leben auf der alten Tabelle –
geteilter Randknoten, Residuum des ersten Stützpunkts und `vel0` würden falsch vorhersagen. Neu:

- Flag Bit 5 (`RESCALE`, Wert 32), `version = 5` (die Version bleibt die höchste, die ein gesetztes Flag verlangt; andere
  Flags sind kombinierbar). Gesetzt in einem Nicht-Schlüsselabschnitt **genau dann**, wenn sich `offset`/`scale` der ersten
  `Σ dims` Skalare (P R S O D) gegen den Vorabschnitt ändern; der Decoder prüft das (Flag ohne Änderung oder Änderung ohne
  Flag → Fehler). Schlüsselabschnitte haben keine Referenz, ihr Flag ist nie gesetzt.
- Der Decoder setzt vor dem Dekodieren `last[g][i][c] := floor((off_alt + q·sc_alt − off_neu) / sc_neu + 0.5)` (IEEE-double,
  in genau dieser Reihenfolge, `off`/`sc` die float32-Kopfwerte; Python = JS) und die Endgeschwindigkeiten `vel := floor(v·sc_alt/sc_neu + 0.5)`;
  geteilter Randknoten, Residuum und `vel0` benutzen diese umgerechneten Werte. Danach läuft alles wie in §4/§8/§9.
- Die **SN-Tabelle** (`offset`/`scale` der letzten `n_sn` Skalare) darf sich innerhalb eines Schlüsselintervalls nicht ändern
  (der SN der weiterlebenden Splats ist Zustand); Encoder und Decoder weisen es ab. Ein Wechsel an einem Schlüsselabschnitt
  ist erlaubt (kein Flag nötig).
- Ohne Tabellenwechsel ist jeder Abschnitt byteidentisch mit v1–v4 (Golden-Tests unverändert, `rd2_long4c`-Reproduktion
  bytegleich). Encoder-Seite: `codec.prep.sections_from_curves(plan=[{qpos, attr_levels, rot_levels, ktol}, …])`
  (`attr_levels` = Schritte über den S/O/D-Bereich, Standard 255; `rot_levels` Standard 127), Tabellen je Abschnitt in
  `stats['section_params']`; Tests `tests/test_falt_rescale.py`.
- Web-Decoder (`web/src/falt.ts`): portiert – `rescaleState` mit den zwei Formeln oben (IEEE-doubles, `Math.floor`, float32-Kopfwerte
  als double), dieselben Prüfungen wie Python (Flag ohne Änderung / Änderung ohne Flag / Flag am Schlüsselabschnitt / SN-Tabelle
  im Intervall); bitgenau getestet mit `synth_rescale*` (Tabelle wechselt je Abschnitt, mit `pred` inkl. `vel0`, Follow, SN_VQ).

## 11 B-Spline-Kurven – Version 6, Flag `BSPLINE`

Flag Bit 6 (`BSPLINE`, Wert 64), `version = 6` (die Version bleibt die höchste, die ein gesetztes Flag verlangt; alle anderen
Flags sind kombinierbar, der Decoder weist Flag ohne Version 6 und umgekehrt ab). Ohne das Flag ist jeder Abschnitt
byteidentisch mit v1–v5 (Golden-Digests aller Versionen unverändert, `tests/test_falt_bspline.py::test_without_flag_nothing_changes`).
**Es ändert sich nur die Bedeutung der Stützpunkt-Ganzzahlen: Kontrollpunkte eines geklemmten kubischen B-Splines statt
Hutfunktions-Knoten.** Bitstrom, Ströme, Kontexte, Quantisierung, geteilter Randknoten, `pred`, `vel0`, `follow`, `RESCALE` sind
unverändert (das Flag gilt für den ganzen Abschnitt, also alle Gruppen; ein Encoder setzt es für den ganzen Strom).

- **Auswertung** (Decoder-Seite, nicht Teil der Entropiecodierung): `u = (t − g0) / max(nframes − 1, 1)` ∈ [0,1], Wert =
  `Σ_j w_j(K, u) · cp_j` mit `w = bspline_weights(K, u)` (`codec/falt.py`, `web/src/bspline.ts`): Grad `p = min(3, K−1)`;
  Knotenvektor `p+1` Nullen, `K−p−1` gleichmäßige innere Knoten `(i+1)/(K−p)`, `p+1` Einsen; Cox–de Boor-Rekursion über
  alle K Basisfunktionen in IEEE-double in genau dieser Operationsreihenfolge (Python = JS, die Gewichte sind **bitgleich**:
  `web/test/bspline.test.ts` vergleicht die 64-Bit-Muster). K=1: konstant, K=2: linear (= Hutbasis), K=3: quadratisch,
  K≥4: kubisch. Geklemmt: `u=0` → Kontrollpunkt 0, `u=1` → Kontrollpunkt K−1 – der geteilte/weiche Randknoten, `State.last`
  und die Endgeschwindigkeit behalten ihre Bedeutung (letzter Kontrollpunkt = Wert am Abschnittsende). Rotation (R) wird nach
  der Summe normiert (nlerp). Partition der Einheit, lokaler Träger (≤ 4 von K Gewichten ≠ 0).
- **Vorhersager bleiben deterministisch über die Kontrollpunkte** (`temporal`, `spatial`, `vel0`, `apply_follow` rechnen auf dem
  Kontrollpolygon wie bisher; ein Kontrollpolygon ist glatt, wenn die Kurve glatt ist). Sie sind nicht auf die Basis abgestimmt –
  eine B-Spline-eigene Vorhersage (z. B. Knoten-Einfügen für K → 2K−1) ist nicht Teil von v6.
- **Encoder-Seite** (`codec.prep.sections_from_curves(basis='bspline'|None)`, `codec.measure --basis auto|hat|bspline`;
  `auto` liest das Feld `basis` der `curves.npz` des Trainers, `E7_BASIS=bspline`): die trainierten Kontrollpunkte werden
  direkt quantisiert (kein Nachfit an Tracks). Die K-Reduktion (`ktol`) verfeinert für jedes K' < K die Kontrollpunkte per
  kleinste Quadrate an den quantisierten K-Kurvenwerten der ganzzahligen Frames (erster und letzter Kontrollpunkt fest) und
  nimmt das kleinste K', dessen dekodierte Kurve in jedem Frame ≤ `ktol` Schritte abweicht (`_reduce_k_bspline`; Hut: Teilnetz
  `(K−1) % (K'−1) == 0`, hier beliebig).
- **Adaptives K** (kein Formatwechsel): das Format trägt K je Splat und Gruppe seit v1 (`k`-Strom); `curves.npz` darf
  `g<i>_kn` (trainiertes K je Splat, Spalten darüber = Füllung) tragen, `prep` reduziert dann je Gruppe gleicher `kn`
  (`_reduce_adaptive`). Trainer: `E7_KADAPT`.
- **Web-Decoder** (`web/src/falt.ts`, `pack.ts`, `shaders.ts`, `bspline.ts`): Version 6 und Flag 64 werden gelesen
  (`Params.bspline`), `pack.ts` setzt Bit 31 im K-Wort des Splat-Headers, die CPU-Referenz `evalSplat` benutzt die
  bitgleichen Gewichte, der WGSL-Auswerte-Pass (`eval_curves`) die lokale Form (NURBS-Buch A2.2, f32; `bsplineLocal` ist ihr
  TS-Spiegel, gleich bis 1e-12 getestet). Der WGSL-Pfad wurde in dieser Arbeit **nicht in einem Browser ausgeführt**.
  Die v3/v4/v5-Flags (pred/follow/rescale) bleiben im TS-Decoder bis zur Portierung durch abgelehnt.
- Tests: `tests/test_falt_bspline.py` (Gewichte, Roundtrip mit/ohne `pred`, geteilter Knoten ja/nein, Version/Flag-Prüfung,
  Golden-Digest v6, K-Reduktion, adaptive Reduktion), `web/test/bspline.test.ts` (Gewichte bitgleich, Dekodierung des
  synthetischen v6-Stroms gleich dem Python-Decoder, Kurvenauswertung gegen das numpy-Orakel).

