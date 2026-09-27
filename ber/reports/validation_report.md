# Validation report (final model, threshold 0.94)

Validation S1 entities: 440,819. Macro F0.5 = **0.97810** (total loss 0.02190).

## By country

| country | entities | F0.5 | share of total loss |
|---|---|---|---|
| India | 176,869 | 0.9694 | 0.0123 |
| US | 263,950 | 0.9840 | 0.0096 |

## By number of true matches (0 = singleton)

| true matches | entities | F0.5 | share of total loss |
|---|---|---|---|
| 0 | 24,506 | 0.9840 | 0.0009 |
| 1 | 23,831 | 0.9219 | 0.0042 |
| 2 | 74,995 | 0.9724 | 0.0047 |
| 3 | 105,947 | 0.9805 | 0.0047 |
| 4 | 97,063 | 0.9837 | 0.0036 |
| 5 | 64,528 | 0.9851 | 0.0022 |
| 6 | 32,538 | 0.9857 | 0.0011 |
| 7 | 12,712 | 0.9857 | 0.0004 |
| 8 | 4,699 | 0.9857 | 0.0002 |

## By name ambiguity

| S1s sharing the exact name | entities | F0.5 | share of total loss |
|---|---|---|---|
| (-inf, 1] | 217,992 | 0.9873 | 0.0063 |
| (1, 2] | 43,765 | 0.9775 | 0.0022 |
| (100, inf] | 17,818 | 0.9716 | 0.0011 |
| (2, 5] | 52,090 | 0.9765 | 0.0028 |
| (20, 100] | 82,502 | 0.9574 | 0.0080 |
| (5, 20] | 26,652 | 0.9750 | 0.0015 |

## By name length

| name length (chars) | entities | F0.5 | share of total loss |
|---|---|---|---|
| (-inf, 5] | 5,522 | 0.9659 | 0.0004 |
| (10, 20] | 258,607 | 0.9761 | 0.0140 |
| (20, 30] | 118,392 | 0.9834 | 0.0045 |
| (30, inf] | 21,410 | 0.9847 | 0.0007 |
| (5, 10] | 36,888 | 0.9733 | 0.0022 |

## By candidate count

| candidates per S1 | entities | F0.5 | share of total loss |
|---|---|---|---|
| (-inf, 10] | 15,217 | 0.9881 | 0.0004 |
| (10, 20] | 122,242 | 0.9841 | 0.0044 |
| (120, inf] | 5,642 | 0.9600 | 0.0005 |
| (20, 40] | 139,849 | 0.9789 | 0.0067 |
| (40, 80] | 127,942 | 0.9740 | 0.0076 |
| (80, 120] | 29,927 | 0.9659 | 0.0023 |

## Error decomposition (pairs)

| item | count |
|---|---|
| true positives | 1,447,707 |
| false positives | 5,355 |
|   of which record belongs to another S1 | 1,237 |
|   of which record belongs to no S1 (distractor) | 4,118 |
|   share of false positives with a blank S2/S3 address | 0.165 |
| false negatives | 77,725 |
|   of which never a candidate (blocking miss) | 53,676 |
|   of which scored below threshold (matcher miss) | 24,049 |
| singleton S1 entities | 24,506 |
| singletons with >=1 false match | 392 |

## Candidate set (what the matcher scores)

| split | S1 | pairs | mean | median | max | reduction ratio | pair recall (validation) |
|---|---|---|---|---|---|---|---|
| train | 2,206,821 | 85,162,046 | 38.6 | 31 | 2,654 | 0.9999963 | 0.9648 |
| test | 1,732,544 | 70,531,037 | 40.7 | 33 | 5,067 | 0.9999959 | - |
