# L1 chance-match calibration on finqa (150 items x 5 probes; probes are gold answers of other items)

| probe magnitude | probes | explained by chance | rate % |
|---|---|---|---|
| decimal, 1 decimals | 60 | 30 | 50.0 |
| decimal, 2 decimals | 35 | 15 | 42.9 |
| decimal, 3+ decimals | 518 | 142 | 27.4 |
| integer 13 to 999 | 53 | 31 | 58.5 |
| integer <= 12 | 9 | 9 | 100.0 |
| integer >= 1000 | 70 | 22 | 31.4 |
| all | 745 | 249 | 33.4 |

Reading: an L1 label on a number in a bucket with a high chance rate is weak evidence of in-text computation; the paper reports L1 with this table beside it and may restrict L1 to buckets under a chosen chance rate.
