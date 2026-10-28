# Simulations project 

An investigation of the modulation of the TL effect by allography in the Arabic language was conducted by creating five datasets which consisted of 478 words of length ranging from five to ten. 'Target' consists of the words as they appear in their original form. The ‘TL-Allog’ dataset includes target words with transposed letters and unchanged allographs. The transposition of one or two letters of target words causes allographs to be altered in the 'TL+Allog' dataset. 'Subl-Allog' comprises target words with substituted letters and unchanged allographs. In the 'Subl+Allog' dataset, one or two letters of target words were substituted such that the allographs were changed. An instance of the various dataset values is shown below:

| Dataset Name | Word |
|---|---|
| Target | الاقتصادية |
| TL-Allog | الاقتصادية |
| TL+Allog | الاقتصادية |
| SubI-Allog | الاقتصادية |
| SubI+Allog | الاقتصادية |


# Arabic Encoding
## Identity Encoding


A **35-dimensional binary vector** that records which Arabic letters are present in a word.

The tokenizer maps each of the 35 Arabic letters to an index (1–35). For each index, the value is `1` if that letter appears in the word, `0` otherwise.

---

## Example: `باب`

The word contains two distinct letters: **ا** (index 6) and **ب** (index 7).

```
Index:  1   2   3   4   5   6   7   8  ...  35
        ء   أ   ؤ   إ   ئ   ا   ب   ت  ...   ي
Value:  0   0   0   0   0   1   1   0  ...   0
                            ↑   ↑
                            ا   ب
```

**Output vector (35 dims):**
```
[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

> Note: duplicates are ignored — ب appears twice in باب but the slot is still just `1`.


## Position Encoding



A **100-dimensional binary vector** that records where each letter appears in the word.

Letters are **sorted alphabetically**, and each letter's position in the original word is encoded as a **10-bit one-hot block** — only the bit at that position is turned on, all others are off.

---

## Example: `الانفصالية`

```
ا  ل  ا  ن  ف  ص  ا  ل  ي  ة
1  2  3  4  5  6  7  8  9  10
```

Unique letters sorted alphabetically, with their positions:

| Letter | Positions |
|---|---|
| ا | 1, 3, 7 |
| ة | 10 |
| ص | 6 |
| ف | 5 |
| ل | 2, 8 |
| ن | 4 |
| ي | 9 |

Each position becomes a 10-bit block with only that bit turned on:

| Letter | Position | 10-bit block |
|---|---|---|
| ا | 1  | `1000000000` |
| ا | 3  | `0010000000` |
| ا | 7  | `0000001000` |
| ة | 10 | `0000000001` |
| ص | 6  | `0000010000` |
| ف | 5  | `0000100000` |
| ل | 2  | `0100000000` |
| ل | 8  | `0000000100` |
| ن | 4  | `0001000000` |
| ي | 9  | `0000000010` |

**Final position vector (100 dims):**
```
[1,0,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0,0,1,
 0,0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0,0, 0,1,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,0,0,
 0,0,0,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,1,0]
```

> This word is 10 letters long, so all 100 dims are used with no padding needed. A 8-letter word, for example, would have its last 20 bits set to `0`.


# Allograph Encoding

A **116-dimensional binary vector** that records which visual form (allograph) of each Arabic letter appears in the word.

Arabic letters change shape depending on their position in the word. The allograph alphabet covers all visual forms of all 35 letters, giving 116 slots total. Each slot is `1` if that specific form appears in the word, `0` otherwise.

---

## The four possible forms

| Form | Example (ب) | When |
|---|---|---|
| Isolated | ب | stands alone |
| Final | ـب | end of word or after non-connector |
| Medial | ـبـ | between two connectors |
| Initial | بـ | start or after non-connector, connects forward |

> Not all letters have four forms. Letters like ا، و، د only have isolated and final (class 2). ء has one form only (class 1).

---

## How it works

The word is first reshaped using `arabic_reshaper`, which converts each letter into its actual visual form. The Keras Tokenizer is then fit on the full allograph alphabet (all forms of all 35 letters). For each allograph slot, the value is `1` if that form appears in the reshaped word, `0` otherwise.

---

## Example: `البرحينية`

After reshaping:
```
ا   ل   بـ  ـر  ـح  ـيـ ـنـ ـيـ ـة
1   2   3   4   5   6   7   8   9
```

> Note: ي appears twice (positions 6 and 8) but both are the same medial form ـيـ, so only one slot fires.

**1s at indexes:**

| Index | Allograph | Letter | Form |
|---|---|---|---|
| 13 | ـا | ا | final |
| 17 | بـ | ب | initial |
| 24 | ـة | ة | final |
| 36 | ـح | ح | medial |
| 46 | ـر | ر | final |
| 96 | لـ | ل | initial |
| 103 | ـنـ | ن | medial |
| 115 | ـيـ | ي | medial |

**Final allograph vector (116 dims):**
```
[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
 0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0]
```

> All absent letters and unused forms remain `0`.


## Output Encoding (One-Hot)

A **350-dimensional vector** that encodes the target word letter by letter.

Each letter is represented as a **35-dim one-hot vector** — one slot per letter in the alphabet, with only the slot for that letter turned on. These are concatenated in word order and zero-padded to 350 dims (35 × 10 max letters).

---

## Example: `الانفصالية`

```
ا   ل   ا   ن   ف   ص   ا   ل   ي   ة
1   2   3   4   5   6   7   8   9   10
```

Each letter gets a 35-dim one-hot block:

| Letter | Position in alphabet | One-hot block (35 dims) |
|---|---|---|
| ا | 6  | `[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]` |
| ل | 29 | `[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]` |
| ... | ... | ... |

All 10 blocks concatenated → **350 dims** with no padding needed for a 10-letter word.





> Note: the one-hot assignment is **reversed** relative to alphabet order — ء gets the last one-hot pattern and ي gets the first. This is an implementation detail that has no effect on the network as long as it is consistent.


# English Words Encoding 

Three encoding schemes are used, each in a **case-sensitive (52-letter)** and **case-insensitive (26-letter)** version. `MAX_LEN = 10` throughout.

---

## 1. Identity vector — "how many of each letter"

A bag-of-letters count, ignoring position and order.

- **Case-sensitive:** 52-length vector (`a`–`z` at indices 0–25, `A`–`Z` at indices 26–51)
- **Case-insensitive:** 26-length vector (word lowercased first)

**Example — "September" (case-sensitive):**
- `S` → 1 (uppercase slot)
- `e`=3, `p`=1, `t`=1, `m`=1, `b`=1, `r`=1 (lowercase slots)

---

## 2. Position vector — "which letter sits in which slot"

A 10×10 grid (flattened to 100 values). Rows = the word's **distinct letters, sorted**; columns = **character position** in the word (1st, 2nd, 3rd...).

**Sort order rule:** letters are sorted by the code's index table, which is `a`–`z` (0–25) **then** `A`–`Z` (26–51). This means **any uppercase letter always sorts after every lowercase letter**, not in normal alphabetical order.

**Example — "nanny" (all lowercase, straightforward):**
- Letters by position: n(1), a(2), n(3), n(4), y(5)
- Distinct letters sorted: a, n, y
- Row `a` → position 2; row `n` → positions 1,3,4; row `y` → position 5

**Example — "goveRnment" (mixed case — the tricky one):**
- Letters by position: g(1) o(2) v(3) e(4) R(5) n(6) m(7) e(8) n(9) t(10)
- Distinct letters sorted **correctly**: e, g, m, n, o, t, v, **R** ← uppercase `R` goes **last**, not alphabetically between `o` and `t`
- Row `e` → positions 4,8; row `g` → 1; row `m` → 7; row `n` → 6,9; row `o` → 2; row `t` → 10; row `v` → 3; row `R` → 5


- **Case-insensitive version:** word is lowercased before encoding, so this ambiguity never arises — sorting is plain alphabetical.

---

## 3. Orthographic vector — "exact letter at exact position"

A flat one-hot vector: alphabet size × `MAX_LEN`. Unlike the position vector, rows/slots correspond to the **full alphabet**, not just the word's distinct letters — so most of the vector is zero for any given word.

- **Case-sensitive:** 52 × 10 = **520**-length vector
- **Case-insensitive:** 26 × 10 = **260**-length vector (word lowercased first)

**Example:** "Same" → length-520 vector, with a 1 at `(position 0, letter 'S')`, `(position 1, letter 'a')`, `(position 2, letter 'm')`, `(position 3, letter 'e')`.

---

## Quick comparison

| Encoding | Captures | Vector length (case-sensitive) | Vector length (case-insensitive) |
|---|---|---|---|
| Identity | letter counts only | 52 | 26 |
| Position | which distinct letter is where, relative to the word | 100 (10×10) | 100 (10×10) |
| Orthographic | exact letter at exact absolute position | 520 (52×10) | 260 (26×10) |

