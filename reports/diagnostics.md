# Learning curve

| fraction | train_macro_f1 | val_macro_f1 | gap |
| --- | --- | --- | --- |
| 0.25 | 1.0 | 0.655 | 0.345 |
| 0.5 | 1.0 | 0.8825000000000001 | 0.11749999999999994 |
| 0.75 | 1.0 | 0.9349999999999999 | 0.06500000000000006 |
| 1.0 | 1.0 | 0.9650000000000001 | 0.03499999999999992 |

# McNemar test

| name_a | name_b | errors_a | errors_b | b01 | b10 | p_value |
| --- | --- | --- | --- | --- | --- | --- |
| tfidf_logreg | tfidf_logreg_balanced | 16 | 14 | 4 | 2 | 0.6875 |

# Novel message check

| text | expected | predicted | correct |
| --- | --- | --- | --- |
| someone drained my wallet overnight without authorisation | fraud-report | fraud-report | True |
| unauthorised party moved my coins to an address I do not recognise | fraud-report | fraud-report | True |
| i think a criminal took over my portfolio and stole everything | fraud-report | fraud-report | True |
| two factor code never arrives so i am locked out | account-access | account-access | True |
| cannot sign in, the verification step keeps failing | account-access | account-access | True |
| i sent money three days ago and it never arrived, want it back | transaction-dispute | transaction-dispute | True |
| charged twice for the same purchase, need a refund | transaction-dispute | transaction-dispute | True |
| what are the fees for converting between currencies | general | general | True |
| do you support customers living in norway | general | general | True |
| how long does identity verification usually take | general | general | True |