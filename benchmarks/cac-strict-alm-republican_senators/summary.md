# CAC-strict Benchmark

- Scorer: `external_scores_csv`
- CAC-strict: 95.2%
- Pairs: 600
- Pairwise accuracy: 95.2%
- Non-tie accuracy: 95.2%
- Tie rate: 0.0%
- Mean margin: 28.591
- Median margin: 19.575
- Train rows: 83303
- Eval rows: 9256
- Eligible eval rows: 8632
- Sampled eval rows: 200

## By Perturbation

| Perturbation | Pass | Fail | Tie | Pairs | Strict Acc. | Non-Tie Acc. | Mean Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adjacent_content_swap | 196 | 4 | 0 | 200 | 98.0% | 98.0% | 22.000 |
| common_token_swap | 175 | 25 | 0 | 200 | 87.5% | 87.5% | 10.675 |
| nonce_substitution | 200 | 0 | 0 | 200 | 100.0% | 100.0% | 53.099 |

## Hard Cases

- 1619019415974182912:common_token_swap: fail margin=-10.852 authentic=-45.006 perturbed=-34.154 text=🚨 I lost my native homeland to a barbaric Communist dictatorship and will do everything I can to ensure my cost… https:/
- 1624867884475228161:common_token_swap: fail margin=-7.021 authentic=-43.842 perturbed=-36.821 text=The CCP's spy efforts should have been immediately shot down. Instead it was allowed to go over three of our most s… htt
- 1621916504001482755:common_token_swap: fail margin=-6.945 authentic=-40.172 perturbed=-33.227 text=Proud to support The BIRTHDAY Act to expand Pell Grant eligibility to high-quality, short-term programs for workers loo…
- 1618436030235017219:common_token_swap: fail margin=-4.470 authentic=-28.596 perturbed=-24.126 text=This goes beyond unthinkable and puts the lives of Americans directly in harm’s way. Our communities are more public… ht
- 1625634042098995206:common_token_swap: fail margin=-4.466 authentic=-40.495 perturbed=-36.030 text=I enjoyed visiting with TN Valley Farmers in Hardin Co. to discuss their vital work finding solutions to coming… https:/
- 1618239767056891904:common_token_swap: fail margin=-3.857 authentic=-25.882 perturbed=-22.025 text=The Biden Administration would rather penalize our military service members for refusing to take the COVID-19 continues…
- 1625204637774057499:common_token_swap: fail margin=-3.670 authentic=-62.662 perturbed=-58.992 text=Scotch Plains Mayor Josh Economy, Deputy Mayor Ellen Zimmerman, Councilman Roc White, and I had the opportunity to… http
- 1617952625550585857:common_token_swap: fail margin=-3.457 authentic=-23.995 perturbed=-20.538 text=🚨I just introduced the CHOICE Act to empower parents. Let's give parents and students the opportunity to take part… http
- 1623069473472147483:common_token_swap: fail margin=-3.129 authentic=-41.196 perturbed=-38.067 text=The American dream is alive and worth fighting for, and my friend @RepCiscomani is a living embodiment of that service… 
- 1626284289997373445:common_token_swap: fail margin=-2.641 authentic=-34.500 perturbed=-31.859 text=Yet again, the FBI has been caught trying to target Americans with conservative beliefs. Withdrawing the tonight… https:
- 1618037693614788608:common_token_swap: fail margin=-2.608 authentic=-21.419 perturbed=-18.811 text=Better! Big business loves big government and big government loves big business. More freedom. Less government.… https:/
- 1624530287966457857:common_token_swap: fail margin=-2.008 authentic=-54.676 perturbed=-52.669 text=Shoutout to our GREAT pilots who shot down the unidentified object over Canadian airspace. We have the most force… https
- 1621515097456738304:common_token_swap: fail margin=-1.951 authentic=-29.744 perturbed=-27.792 text=Protecting American farmland is a national security priority. I hear from many farmers in Central WA who are come… https
- 1620109691119607808:common_token_swap: fail margin=-1.806 authentic=-35.246 perturbed=-33.441 text=Despite President Biden declaring "the pandemic is over" in September, his Admin extended the Public Health Tune… https:
- 1623723841812824066:common_token_swap: fail margin=-1.740 authentic=-23.411 perturbed=-21.671 text=🚨🚨TODAY🚨🚨   @Weaponization is pulling back the curtain to EXPOSE the FBI’s unconstitutional censorship of freedom… https
- 1622614572375977984:common_token_swap: fail margin=-1.658 authentic=-42.447 perturbed=-40.789 text=We have two forward economic threats: a risk of default &amp; a debt to GDP ratio that exceeds 100%, which has nev… http
- 1624159784415944729:common_token_swap: fail margin=-1.627 authentic=-19.020 perturbed=-17.393 text=🚨 #SOTU Fact Check 🚨   From inflation to immigration, President Biden’s State of the Union address relied on cost… https
- 1617928171885854720:common_token_swap: fail margin=-1.570 authentic=-32.935 perturbed=-31.365 text=I enjoyed hearing from members of our agricultural community, including young entrepreneurs, at BRAVE last week. R… http
- 1625246455400132629:adjacent_content_swap: fail margin=-1.458 authentic=-23.555 perturbed=-22.097 text=We're supposed to be the most militarily and technologically advanced nation in the world.  Either this statement… https
- 1620851064844218369:adjacent_content_swap: fail margin=-1.152 authentic=-48.532 perturbed=-47.380 text=Es excepcional unirme esta mañana a mis colegas de la Hispana Conferencia del Congreso para discutir nuestra misión… htt
