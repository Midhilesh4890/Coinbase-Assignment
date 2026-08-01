# Learning curve

Pipeline: tfidf_logreg_tuned

| fraction | train_macro_f1 | val_macro_f1 | gap |
| --- | --- | --- | --- |
| 0.2500 | 1.0000 | 0.6975 | 0.3025 |
| 0.5000 | 1.0000 | 0.9050 | 0.0950 |
| 0.7500 | 1.0000 | 0.9450 | 0.0550 |
| 1.0000 | 1.0000 | 0.9850 | 0.0150 |

# McNemar test

| name_a | name_b | errors_a | errors_b | b01 | b10 | p_value |
| --- | --- | --- | --- | --- | --- | --- |
| tfidf_logreg | tfidf_logreg_balanced | 16 | 14 | 4 | 2 | 0.6875 |
| tfidf_logreg_balanced | tfidf_logreg_tuned | 14 | 6 | 8 | 0 | 0.0078 |

# Novel message check

Pipeline: tfidf_logreg_tuned

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