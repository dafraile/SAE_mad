# NL ↔ NF accuracy-gap decomposition

Triggered by reviewer concern that §4.2's deferral-driven framing for 12B may be incorrect. This script does the proper case-level decomposition.

## gemma-3-4b-it

- n = 60
- NL accuracy: **55.0%**
- NF accuracy (both 4-way judges correct): **71.7%**
- **Gap (NL − NF): -16.7 pp**

### Stratum counts
- `both_right      `: 29
- `NF_only_right   `: 14
- `NL_only_right   `: 1
- `both_wrong      `: 12
- `judges_disagree `: 4

### Deferral location
- Unanimous DEFERRED total: **0**
- ...of which live in `both_right`        (counted correct under 4-way): **0**
- ...of which live in `NL_only_right`     (contribute to NL > NF gap):   **0**

### Gap-driving stratum: NL_only_right (n=1)
- Unanimous DEFERRED: 0 / 1
- Either-judge DEFERRED: 0 / 1
- Adjacency of NL vs NF letter (when judges agree):
  - adjacent: 1

  Per-case:
  | case | gold | NL | NF gpt | NF cla | adj | 5-way unanim |
  |---|---|---|---|---|---|---|
  | F25 | C | C | D | D | adjacent |  |

### Counter-stratum: NF_only_right (n=14)
- Unanimous DEFERRED: 0 / 14
- Adjacency of NL vs NF letter (when judges agree):
  - adjacent: 14

  Per-case:
  | case | gold | NL | NF gpt | NF cla | adj | 5-way unanim |
  |---|---|---|---|---|---|---|
  | E3 | C | B | C | C | adjacent |  |
  | E4 | C | B | C | C | adjacent |  |
  | E10 | C/D | B | C | C | adjacent |  |
  | E11 | C/D | B | C | C | adjacent |  |
  | E22 | C/D | B | C | C | adjacent |  |
  | E25 | C | B | C | C | adjacent |  |
  | F1 | C/D | B | C | C | adjacent |  |
  | F3 | C | B | C | C | adjacent |  |
  | F4 | C | B | C | C | adjacent |  |
  | F10 | C/D | B | C | C | adjacent |  |
  | F19 | B | A | B | B | adjacent |  |
  | MH1 | C | B | C | C | adjacent |  |
  | NH1 | C | B | C | C | adjacent |  |
  | NH3 | C | B | C | C | adjacent |  |

## gemma-3-12b-it

- n = 60
- NL accuracy: **81.7%**
- NF accuracy (both 4-way judges correct): **71.7%**
- **Gap (NL − NF): +10.0 pp**

### Stratum counts
- `both_right      `: 43
- `NF_only_right   `: 0
- `NL_only_right   `: 6
- `both_wrong      `: 11
- `judges_disagree `: 0

### Deferral location
- Unanimous DEFERRED total: **4**
- ...of which live in `both_right`        (counted correct under 4-way): **4**
- ...of which live in `NL_only_right`     (contribute to NL > NF gap):   **0**

### Gap-driving stratum: NL_only_right (n=6)
- Unanimous DEFERRED: 0 / 6
- Either-judge DEFERRED: 1 / 6
- Adjacency of NL vs NF letter (when judges agree):
  - adjacent: 5
  - non_adjacent: 1

  Per-case:
  | case | gold | NL | NF gpt | NF cla | adj | 5-way unanim |
  |---|---|---|---|---|---|---|
  | E19 | B | B | C | C | adjacent |  |
  | F3 | C | C | B | B | adjacent |  |
  | F7 | C/D | C | A | A | non_adjacent |  |
  | F11 | C/D | C | B | B | adjacent |  |
  | F13 | D | D | C | C | adjacent |  |
  | NH3 | C | C | D | D | adjacent |  |

### Counter-stratum: NF_only_right (n=0)
- Unanimous DEFERRED: 0 / 0
- Adjacency of NL vs NF letter (when judges agree):

  Per-case:
  | case | gold | NL | NF gpt | NF cla | adj | 5-way unanim |
  |---|---|---|---|---|---|---|

## qwen3-8b

- n = 60
- NL accuracy: **75.0%**
- NF accuracy (both 4-way judges correct): **68.3%**
- **Gap (NL − NF): +6.7 pp**

### Stratum counts
- `both_right      `: 35
- `NF_only_right   `: 6
- `NL_only_right   `: 8
- `both_wrong      `: 6
- `judges_disagree `: 5

### Deferral location
- Unanimous DEFERRED total: **2**
- ...of which live in `both_right`        (counted correct under 4-way): **0**
- ...of which live in `NL_only_right`     (contribute to NL > NF gap):   **0**

### Gap-driving stratum: NL_only_right (n=8)
- Unanimous DEFERRED: 0 / 8
- Either-judge DEFERRED: 0 / 8
- Adjacency of NL vs NF letter (when judges agree):
  - adjacent: 8

  Per-case:
  | case | gold | NL | NF gpt | NF cla | adj | 5-way unanim |
  |---|---|---|---|---|---|---|
  | E3 | C | C | B | B | adjacent |  |
  | E4 | C | C | B | B | adjacent |  |
  | E8 | A | A | B | B | adjacent |  |
  | E17 | A | A | B | B | adjacent |  |
  | F1 | C/D | C | B | B | adjacent |  |
  | F4 | C | C | B | B | adjacent |  |
  | F11 | C/D | C | B | B | adjacent |  |
  | F25 | C | C | D | D | adjacent |  |

### Counter-stratum: NF_only_right (n=6)
- Unanimous DEFERRED: 0 / 6
- Adjacency of NL vs NF letter (when judges agree):
  - non_adjacent: 5
  - adjacent: 1

  Per-case:
  | case | gold | NL | NF gpt | NF cla | adj | 5-way unanim |
  |---|---|---|---|---|---|---|
  | E6 | B/C | A | C | C | non_adjacent |  |
  | E7 | C/D | A | C | C | non_adjacent |  |
  | F6 | B/C | A | C | C | non_adjacent |  |
  | F7 | C/D | A | C | C | non_adjacent |  |
  | F10 | C/D | A | C | C | non_adjacent |  |
  | F24 | B | A | B | B | adjacent |  |

---

## Bottom line

The 12B NL → NF accuracy gap is **NOT** driven by deferral: all 4 unanimous DEFERRED cases at 12B happen to flatten to gold-compatible letters under 4-way scoring (so they live in `both_right`, not in `NL_only_right`) and contribute zero to the accuracy gap. The gap is driven by **adjacent miscalibration**: 5/6 NL_only_right cases at 12B have NL on the gold letter and NF on a one-step-adjacent letter.

Symmetrically, the 4B NF → NL gap is driven by the inverse pattern: 14/14 NF_only_right cases at 4B have NL one step *below* the gold (most commonly B-instead-of-C) while NF judges agree on the gold letter.

Deferral is a real phenomenon (4/60 unanimous at 12B, 2/60 at Qwen, 0/60 at 4B) but it is a *separate* benchmark-adequacy concern about the A/B/C/D label space, not the cause of the measured accuracy inversion.