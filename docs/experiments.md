- lgbmclass benchmark numerical features raw
- log reg: med imputer robust scalar + features numerical raw
- log reg: med imputer and indicators robust scalar + features numerical raw
- log reg: med imputer and indicators robust scalar + features numerical raw + C=0.01

stay with lgbmclass

add fetaures 1:
- mean ret 5 et 20, ret std 5 et 20, ret 1 / ret std 20, win rate, ret std 5 / ret std 20
- ablation win rate

garder win rate

- add volume fts: mean 5 20 et std 5 20
garder volume fts pour l'instant
- tests stability seed a peu pres coirrect donc on garde 53 fts

- look at feature importance
- norm ret ratio 1 20 ablation: la feature la plus importante dans les arbres n’apporte ici qu’un petit avantage supplémentaire en validation

fetaures v3:
- same-date allocation averages and deviations for recent returns: On conserve les 53 features comme référence, pas de motivation a les ajoputer

- come back to fts v2 et add categories features
- add cat_smooth for categories regularization 


plus tard intercation avec alloc, group et numerical categories from group et alloc