# CAC-strict Benchmark

- Scorer: `external_scores_csv`
- CAC-strict: 95.7%
- Pairs: 600
- Pairwise accuracy: 95.7%
- Non-tie accuracy: 95.7%
- Tie rate: 0.0%
- Mean margin: 26.904
- Median margin: 18.926
- Train rows: 88076
- Eval rows: 9787
- Eligible eval rows: 9380
- Sampled eval rows: 200

## By Perturbation

| Perturbation | Pass | Fail | Tie | Pairs | Strict Acc. | Non-Tie Acc. | Mean Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adjacent_content_swap | 195 | 5 | 0 | 200 | 97.5% | 97.5% | 21.440 |
| common_token_swap | 179 | 21 | 0 | 200 | 89.5% | 89.5% | 9.608 |
| nonce_substitution | 200 | 0 | 0 | 200 | 100.0% | 100.0% | 49.663 |

## Hard Cases

- 1625276205233471488:common_token_swap: fail margin=-16.071 authentic=-72.386 perturbed=-56.315 text=Congrats to Leaders Green and Saucy Downs on their performances at the Black History Month kickoff event at the West… ht
- 1624060999157284864:common_token_swap: fail margin=-9.119 authentic=-58.708 perturbed=-49.589 text=INVEST payments have been treated as nontaxable income for decades. IRS telling Coloradans to hold off doing their… http
- 1620542715586019329:common_token_swap: fail margin=-8.044 authentic=-49.947 perturbed=-41.903 text=We need to be energy independent. But we shouldn’t swap foreign oligarchs for domestic oil barons, or trade Republicans…
- 1624530718075547648:common_token_swap: fail margin=-6.059 authentic=-33.694 perturbed=-27.635 text=A big thank you to everyone who made valentines for local Veterans, including Girl Scout Troop 43826! I look around… htt
- 1622664558857920514:common_token_swap: fail margin=-6.044 authentic=-40.152 perturbed=-34.108 text=Mr. Cory Program, Kalama High School's career &amp; technical education teacher, is opening up so many doors for student
- 1627368416595312641:common_token_swap: fail margin=-5.666 authentic=-77.004 perturbed=-71.338 text=I made two stops on my Farm Bill listening tour in Colleagues County, both on farms with generations of family represent
- 1624846801240002561:common_token_swap: fail margin=-4.934 authentic=-60.357 perturbed=-55.423 text=I am ‘paws-passing’ thrilled that Moose, a rescue dog from East Hartford, will be playing in the big leagues on the… htt
- 1620509727389097995:common_token_swap: fail margin=-4.552 authentic=-42.431 perturbed=-37.878 text=Representative Danny K. Davis send Kudos to Secretary of Defense Lloyd Austin for the $90 million contract to Resources…
- 1623157925408645121:common_token_swap: fail margin=-4.113 authentic=-28.801 perturbed=-24.688 text=The opioid epidemic has reached across our country and into our communities in California.   Let me be clear: without… h
- 1618780883414315008:common_token_swap: fail margin=-3.858 authentic=-67.438 perturbed=-63.579 text=Huge and welcome news! The service waters wilderness is the most visited wilderness area in the country and should… http
- 1623147925625884675:common_token_swap: fail margin=-2.879 authentic=-21.544 perturbed=-18.666 text=Big Pharma's inflated drug prices have prevented countless Arizonans from getting the care they need. That's high… https
- 1621584313841061888:common_token_swap: fail margin=-2.544 authentic=-30.219 perturbed=-27.675 text=Michigan farmers work hard to grow the food &amp; fuel America relies on. I recently spoke w/ the Bay County Farm Worker
- 1618780883414315008:adjacent_content_swap: fail margin=-2.259 authentic=-67.438 perturbed=-65.179 text=Huge and welcome news! The waters boundary wilderness is the most visited wilderness area in the country and should… htt
- 1623501940243185666:common_token_swap: fail margin=-2.173 authentic=-50.550 perturbed=-48.377 text=Yesterday's State of the Union was one of optimism. Sharing this experience with my constituent, Hope Council… https://t
- 1619000815284523010:adjacent_content_swap: fail margin=-2.087 authentic=-36.446 perturbed=-34.359 text=Since before I was born, we’ve allowed discriminatory and antiquated ideas – instead of science – to determine who… http
- 1620556099286876161:common_token_swap: fail margin=-2.061 authentic=-43.350 perturbed=-41.288 text=🚨🚨 Bill Drop Alert 🚨🚨    Today Congresswoman Better-McCormick introduced the Healthy Foundations for Homeless Ve… https:
- 1625179965334470680:common_token_swap: fail margin=-1.955 authentic=-20.040 perturbed=-18.085 text=#SocialSecurity &amp; #Medicare are a lifeline for millions of Americans. They are EARNED benefits paid into best… https
- 1618349439416164352:common_token_swap: fail margin=-1.758 authentic=-18.059 perturbed=-16.301 text=Last night, I got to meet @SecVilsack with my colleagues in the @HispanicCaucus. I invited him to visit the Still… https
- 1618380665782931456:common_token_swap: fail margin=-1.675 authentic=-13.320 perturbed=-11.645 text=Thanks to the #InflationReductionAct, Seniors no longer have to pay more than $35 a month for insulin, saving legislatio
- 1619047943172136961:common_token_swap: fail margin=-1.076 authentic=-53.151 perturbed=-52.075 text=Another day. Another shooting. Another community month.  True freedom is getting to live without fear of gun… https://t.
