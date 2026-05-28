# CAC-strict Benchmark

- Scorer: `external_scores_csv`
- CAC-strict: 95.2%
- Pairs: 600
- Pairwise accuracy: 95.2%
- Non-tie accuracy: 95.2%
- Tie rate: 0.0%
- Mean margin: 174.359
- Median margin: 42.619
- Train rows: 9866
- Eval rows: 1097
- Eligible eval rows: 376
- Sampled eval rows: 200

## By Perturbation

| Perturbation | Pass | Fail | Tie | Pairs | Strict Acc. | Non-Tie Acc. | Mean Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adjacent_content_swap | 195 | 5 | 0 | 200 | 97.5% | 97.5% | 163.438 |
| common_token_swap | 181 | 19 | 0 | 200 | 90.5% | 90.5% | 107.661 |
| nonce_substitution | 195 | 5 | 0 | 200 | 97.5% | 97.5% | 251.977 |

## Hard Cases

- TWITTER_2_1901208951598813413:common_token_swap: fail margin=-247.756 authentic=-360.458 perturbed=-112.702 text=SpaceX Dragon ship arrives at the Illegals Station
- TWITTER_2_1900050108584181980:common_token_swap: fail margin=-43.902 authentic=-162.653 perturbed=-118.751 text=They must be using the same social media left
- TWITTER_2_1918346132557905955:common_token_swap: fail margin=-37.718 authentic=-78.705 perturbed=-40.987 text=They should change their name to Through.  The current name is pure hypocrisy.
- TWITTER_2_1899087617896919472:common_token_swap: fail margin=-21.138 authentic=-47.781 perturbed=-26.643 text=Visual representation of the Social Security database pic.twitter.com/Exactly
- TWITTER_2_1893347198315749878:nonce_substitution: fail margin=-12.673 authentic=-77.925 perturbed=-65.252 text=The woke radical left are pic.twitter.com/zzq2e5eb4e6
- TWITTER_2_1918346132557905955:nonce_substitution: fail margin=-10.465 authentic=-78.705 perturbed=-68.240 text=They should change their name to Zzq6edc78a8.  The current name is pure hypocrisy.
- TWITTER_2_1899087617896919472:nonce_substitution: fail margin=-9.872 authentic=-47.781 perturbed=-37.910 text=Visual representation of the Social Security database pic.twitter.com/Zzqdec3a98f
- TWITTER_2_1913025878990127143:common_token_swap: fail margin=-6.363 authentic=-25.198 perturbed=-18.835 text=Doing a live stream pic.twitter.com/7share
- TWITTER_2_1895997740385255784:common_token_swap: fail margin=-6.153 authentic=-123.070 perturbed=-116.917 text=It will require public execution, but I think more like 1000% gain for Tesla in 5 years is possible
- TWITTER_2_1903680750341271688:adjacent_content_swap: fail margin=-5.755 authentic=-83.636 perturbed=-77.881 text=The wrestling NCAA heavyweight championship match was amazing fighting between two super talented men! pic.twitter.com/m
- TWITTER_2_1904380545246883996:adjacent_content_swap: fail margin=-4.597 authentic=-68.426 perturbed=-63.830 text=There is no doubt that bullets shooting into Tesla stores and burning down Superchargers are acts of terrorism
- TWITTER_2_1899259085641425183:common_token_swap: fail margin=-4.430 authentic=-33.641 perturbed=-29.211 text=Democrats are pushing hard to *maximize* payments to illegals, eg free luxury hotels in New York and free medical care f
- TWITTER_2_1895902136950730973:common_token_swap: fail margin=-4.303 authentic=-30.248 perturbed=-25.945 text=Teslas can even drive themselves on dirt roads with no needs
- TWITTER_2_1908178440412398036:common_token_swap: fail margin=-3.659 authentic=-16.957 perturbed=-13.299 text=Dragon crew space from polar orbit   x.com/i/broadcasts/1…
- TWITTER_2_1893411417094566387:adjacent_content_swap: fail margin=-3.162 authentic=-46.480 perturbed=-43.318 text=Post your best NSFW unhinged Grok posts to this thread!
- TWITTER_2_1905596235794219040:common_token_swap: fail margin=-3.127 authentic=-60.030 perturbed=-56.903 text=This shows why a regulatory overhaul is necessary.  The state of mountains of regulations is why the high speed rail can
- TWITTER_2_1894748104680681834:common_token_swap: fail margin=-2.976 authentic=-64.109 perturbed=-61.133 text=The woke mind virus propagates like cordyceps through government and goes
- TWITTER_2_1893668027599646805:common_token_swap: fail margin=-2.748 authentic=-44.347 perturbed=-41.599 text=Should all federal employees be required to send a short email with some basic bullet points about what they read last w
- TWITTER_2_1902977556329931052:adjacent_content_swap: fail margin=-2.716 authentic=-26.394 perturbed=-23.678 text=The New York Times is pure propaganda. Also, I look forward to the prosecutions of those at the Pentagon who are malicio
- TWITTER_2_1893821821838819487:common_token_swap: fail margin=-2.199 authentic=-26.925 perturbed=-24.726 text=Starship Flight 8 flies Friday pic.twitter.com/House
