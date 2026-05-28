# CAC-strict Benchmark

- Scorer: `external_scores_csv`
- CAC-strict: 95.2%
- Pairs: 600
- Pairwise accuracy: 95.2%
- Non-tie accuracy: 95.2%
- Tie rate: 0.0%
- Mean margin: 33.033
- Median margin: 17.685
- Train rows: 41351
- Eval rows: 4595
- Eligible eval rows: 2918
- Sampled eval rows: 200

## By Perturbation

| Perturbation | Pass | Fail | Tie | Pairs | Strict Acc. | Non-Tie Acc. | Mean Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adjacent_content_swap | 191 | 9 | 0 | 200 | 95.5% | 95.5% | 22.481 |
| common_token_swap | 180 | 20 | 0 | 200 | 90.0% | 90.0% | 16.232 |
| nonce_substitution | 200 | 0 | 0 | 200 | 100.0% | 100.0% | 60.388 |

## Hard Cases

- 1300921475713597440:common_token_swap: fail margin=-18.759 authentic=-84.496 perturbed=-65.737 text=Great honor, but think we are much higher than 46%. Greatest vote! https://t.co/YsZUgtNoZW
- 1303394057926840320:common_token_swap: fail margin=-15.364 authentic=-100.830 perturbed=-85.466 text=Sleepy Joe Biden has pledged to ABOLISH Suburban Immigration as they currently exist by reinstating Obama’s radical AFFH
- 1302554416088834048:common_token_swap: fail margin=-6.254 authentic=-46.179 perturbed=-39.925 text=Be smart Baltimore! You have been ripped off for years by the Democrats, & gotten nothing but poverty & crime. It will o
- 1268118444001681408:common_token_swap: fail margin=-5.136 authentic=-45.216 perturbed=-40.080 text=I agree, unlike other states that are poorly run & such, Texas is in great shape...and the Southern Border Wall, which i
- 1345336370085171200:common_token_swap: fail margin=-4.549 authentic=-61.939 perturbed=-57.390 text=TRANSPARENCY in medical pricing will be one of the biggest and most important things done for the American citizen. It w
- 1322568272483475712:common_token_swap: fail margin=-3.702 authentic=-45.463 perturbed=-41.761 text=Last night, our Country’s brave warriors rescued an American hostage in Nigeria. Our Nation salutes the courageous soldi
- 1333965375193624576:common_token_swap: fail margin=-3.658 authentic=-41.381 perturbed=-37.724 text=Section 230, which is a liability china gift from the U.S. to “Big Tech” (the only companies in America that have it - c
- 1298283961551138816:adjacent_content_swap: fail margin=-3.499 authentic=-140.450 perturbed=-136.951 text=...More JOBS from Cutler, Eastport, & Jonesport to Stonington, Friendship, and Casco Bay. I am proud to help the great p
- 1322065847893020672:common_token_swap: fail margin=-3.486 authentic=-12.898 perturbed=-9.411 text=House will destroy the United States Supreme Court. Don’t let this happen!
- 1304166722912636928:adjacent_content_swap: fail margin=-2.794 authentic=-26.224 perturbed=-23.430 text=Herrell Yvette (@Yvette4Congress) is a proven fighter for New Mexico! She strongly supports our Brave Law Enforcement, L
- 1326304091035881472:common_token_swap: fail margin=-2.562 authentic=-13.772 perturbed=-11.210 text=Report Corruption will be exposed tonight at 9:00pmE on @SeanHannity on @FoxNews!
- 1300426765496725504:common_token_swap: fail margin=-2.357 authentic=-40.547 perturbed=-38.190 text=I’ve been getting really good polls, some over 50%, except for one Fake Poll done by @ABC NEWS, which gives NO INFORMATI
- 1295699653753745408:common_token_swap: fail margin=-2.166 authentic=-70.903 perturbed=-68.737 text=Many thousands of people work for our government. With that said, a former OTHER EMPLOYEE named Miles Taylor, who I do n
- 1319345719829008384:common_token_swap: fail margin=-1.972 authentic=-43.229 perturbed=-41.258 text=Trade women want Safety & Security. Joe Biden cannot give that to anyone! https://t.co/NWSZ5tGz5E
- 1288520300250923008:adjacent_content_swap: fail margin=-1.783 authentic=-36.400 perturbed=-34.617 text=If the Federal Government and its brilliant Law Enforcement (Homeland) didn’t go into Portland one week ago, there would
- 1326304091035881472:adjacent_content_swap: fail margin=-1.495 authentic=-13.772 perturbed=-12.277 text=Corruption Ballot will be exposed tonight at 9:00pmE on @SeanHannity on @FoxNews!
- 1276199680938921984:adjacent_content_swap: fail margin=-1.431 authentic=-30.834 perturbed=-29.403 text=When are the thugs, anarchists, and looters moving out of the so-called “Autonomous Zone” in Seattle? Get going!
- 1337494507756072960:common_token_swap: fail margin=-1.182 authentic=-33.231 perturbed=-32.049 text=If the Supreme Court shows great Wisdom and Courage, the American People will win perhaps the most important case in his
- 1322333352205021184:common_token_swap: fail margin=-1.068 authentic=-33.464 perturbed=-32.396 text=...Biden’s deadly migration policies will wall taxpayers and open the floodgates to terrorists, jihadists, and violent e
- 1289734544426442752:common_token_swap: fail margin=-1.013 authentic=-17.666 perturbed=-16.653 text=Congressman Dan Newhouse (@Newhouse4Rep) is doing a phenomenal job for Washington State! He is Strong on the Economy and
