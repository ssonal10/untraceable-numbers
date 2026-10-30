# L1 chance-match calibration on tatqa (150 items x 5 probes; probes are gold answers of other items)

| probe magnitude | probes | explained by chance | rate % |
|---|---|---|---|
| decimal, 1 decimals | 147 | 66 | 44.9 |
| decimal, 2 decimals | 277 | 92 | 33.2 |
| integer 13 to 999 | 109 | 59 | 54.1 |
| integer <= 12 | 10 | 10 | 100.0 |
| integer >= 1000 | 179 | 37 | 20.7 |
| all | 722 | 264 | 36.6 |

Reading: an L1 label on a number in a bucket with a high chance rate is weak evidence of in-text computation; the paper reports L1 with this table beside it and may restrict L1 to buckets under a chosen chance rate.
