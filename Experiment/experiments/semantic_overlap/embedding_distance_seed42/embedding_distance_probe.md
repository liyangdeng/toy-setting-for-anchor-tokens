# Semantic Overlap Token Embedding Distance Probe

This probe uses model input token embeddings and cosine distance.

Pairs are true CJK-Hiragana artificial-token pairs that share the same synset, and the synset must appear as a `source` or `target` endpoint in `data/semantic_backbones/edges_adj.json`.

Graph endpoint nodes used for filtering: 2042. Valid graph-token pairs present in every model: 994.

## Distance Summary

| Group | Condition | Overlap | Mean cosine distance | SD | Min | Max | Pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Close at 100% | `overlap_000` | 0% | 0.7534 | 0.1258 | 0.4927 | 0.9995 | 30 |
| Close at 100% | `overlap_025` | 25% | 0.7093 | 0.1406 | 0.3944 | 0.9794 | 30 |
| Close at 100% | `overlap_050` | 50% | 0.6401 | 0.1193 | 0.3817 | 0.9466 | 30 |
| Close at 100% | `overlap_075` | 75% | 0.6993 | 0.1704 | 0.3798 | 1.0863 | 30 |
| Close at 100% | `overlap_100` | 100% | 0.3183 | 0.0168 | 0.2712 | 0.3413 | 30 |
| Random graph true pairs | `overlap_000` | 0% | 0.7638 | 0.1396 | 0.5140 | 1.1371 | 30 |
| Random graph true pairs | `overlap_025` | 25% | 0.7568 | 0.1481 | 0.4403 | 1.0597 | 30 |
| Random graph true pairs | `overlap_050` | 50% | 0.6913 | 0.1298 | 0.4821 | 0.9354 | 30 |
| Random graph true pairs | `overlap_075` | 75% | 0.6445 | 0.1441 | 0.4295 | 0.9970 | 30 |
| Random graph true pairs | `overlap_100` | 100% | 0.5004 | 0.0986 | 0.3642 | 0.7413 | 30 |
| Far at 100% | `overlap_000` | 0% | 0.8527 | 0.1217 | 0.6309 | 1.0508 | 30 |
| Far at 100% | `overlap_025` | 25% | 0.7727 | 0.1315 | 0.4864 | 1.0085 | 30 |
| Far at 100% | `overlap_050` | 50% | 0.7363 | 0.1637 | 0.4873 | 1.1382 | 30 |
| Far at 100% | `overlap_075` | 75% | 0.6786 | 0.1277 | 0.4240 | 0.8815 | 30 |
| Far at 100% | `overlap_100` | 100% | 0.8529 | 0.0592 | 0.7948 | 1.0466 | 30 |

## Selected Graph Pairs

| Group | Rank | Synset | Lemma | CJK token | Hiragana token | Distance at 100% |
| --- | ---: | --- | --- | --- | --- | ---: |
| Close at 100% | 1 | `nettle.n.01` | nettle | 艑姭顑 | てどがら | 0.2712 |
| Close at 100% | 2 | `muscular_structure.n.01` | muscular structure | 欋芋餵咝 | てしふ | 0.2824 |
| Close at 100% | 3 | `conceptnet:colour:1346` | colour | 鎛伭 | らせほゖご | 0.2954 |
| Close at 100% | 4 | `wolfsbane.n.01` | wolfsbane | 鋸喼 | ぺは | 0.2980 |
| Close at 100% | 5 | `anatomist.n.01` | anatomist | 鏏帔亡淂 | ざゆけぁだご | 0.3018 |
| Close at 100% | 6 | `deliveryman.n.01` | deliveryman | 脕 | ゖむざ | 0.3049 |
| Close at 100% | 7 | `conceptnet:injury:1921` | injury | 欻抉勣 | なぼんご | 0.3070 |
| Close at 100% | 8 | `appraiser.n.01` | appraiser | 鯗咁鄦 | ぷた | 0.3100 |
| Close at 100% | 9 | `second_fiddle.n.01` | second fiddle | 閍褻喑錖 | ぐっゔ | 0.3102 |
| Close at 100% | 10 | `salesman.n.01` | salesman | 鴢堚塏 | ずじぞゐ | 0.3168 |
| Close at 100% | 11 | `conceptnet:sea:1263` | sea | 儙壎澠 | まがこぺ | 0.3172 |
| Close at 100% | 12 | `ram.n.05` | ram | 嚴癳瘞鄜 | だり | 0.3183 |
| Close at 100% | 13 | `conceptnet:castle:1301` | castle | 裵蛹箿滦 | ふのなぜ | 0.3189 |
| Close at 100% | 14 | `salesperson.n.01` | salesperson | 跦敖 | ぁがぺわの | 0.3195 |
| Close at 100% | 15 | `double_knit.n.01` | double knit | 釃苐 | とおいそ | 0.3205 |
| Close at 100% | 16 | `conceptnet:hammer:1463` | hammer | 蕢喩阐鶮 | えら | 0.3212 |
| Close at 100% | 17 | `celestial_point.n.01` | celestial point | 鄐辝亼 | たべ | 0.3215 |
| Close at 100% | 18 | `specialist.n.01` | specialist | 橇齦窌 | すぃそ | 0.3217 |
| Close at 100% | 19 | `igbo.n.01` | igbo | 雁摯鳊忴 | れほが | 0.3246 |
| Close at 100% | 20 | `conceptnet:fly:1244` | fly | 凄窰墬 | ぶもゐぎ | 0.3246 |
| Close at 100% | 21 | `aconite.n.01` | aconite | 陝氪 | みょた | 0.3249 |
| Close at 100% | 22 | `alga.n.01` | alga | 裷多 | ぉうろ | 0.3250 |
| Close at 100% | 23 | `tapeworm.n.01` | tapeworm | 緂嘂微 | べぜ | 0.3285 |
| Close at 100% | 24 | `conceptnet:astrology:1434` | astrology | 嗂翂徸 | たば | 0.3341 |
| Close at 100% | 25 | `antacid.n.01` | antacid | 封 | くきとぴ | 0.3341 |
| Close at 100% | 26 | `critic.n.02` | critic | 矌稹 | しぼお | 0.3376 |
| Close at 100% | 27 | `jellyfish.n.02` | jellyfish | 緘脪卷 | おな | 0.3378 |
| Close at 100% | 28 | `conceptnet:sky:1400` | sky | 凨拜 | じぺて | 0.3403 |
| Close at 100% | 29 | `hairdresser.n.01` | hairdresser | 團磺臱 | ゔが | 0.3405 |
| Close at 100% | 30 | `evaluator.n.01` | evaluator | 纑傣罕 | をせ | 0.3413 |
| Random graph true pairs | 1 | `houseplant.n.01` | houseplant | 鄱詺塄 | ゎぶ | 0.5765 |
| Random graph true pairs | 2 | `soup.n.01` | soup | 慗変睗冂 | ぃぎゎ | 0.4097 |
| Random graph true pairs | 3 | `conceptnet:boring:1627` | boring | 鐕磬 | ゅる | 0.3642 |
| Random graph true pairs | 4 | `dipterous_insect.n.01` | dipterous insect | 烞篪 | にえ | 0.6166 |
| Random graph true pairs | 5 | `sleeve.n.01` | sleeve | 甊熿逬 | はやじ | 0.4597 |
| Random graph true pairs | 6 | `conceptnet:flat:1295` | flat | 勤 | をゐろ | 0.4515 |
| Random graph true pairs | 7 | `conceptnet:fishing:1460` | fishing | 澊邲 | ごひづ | 0.4465 |
| Random graph true pairs | 8 | `lap.n.04` | lap | 揹陜睐懈 | ふうあれ | 0.4208 |
| Random graph true pairs | 9 | `conceptnet:eat:1216` | eat | 哩鿼鏒碯 | をゆに | 0.6146 |
| Random graph true pairs | 10 | `jersey.n.04` | jersey | 庛疆恀 | ゕかや | 0.4074 |
| Random graph true pairs | 11 | `system.n.06` | system | 槼諉 | ぽをっ | 0.5875 |
| Random graph true pairs | 12 | `appetizer.n.01` | appetizer | 洰白釢 | ありぺ | 0.6160 |
| Random graph true pairs | 13 | `isogram.n.01` | isogram | 红親 | ほちせこか | 0.7413 |
| Random graph true pairs | 14 | `conceptnet:constant:1546` | constant | 棦柭仾鿡 | ぁをゆ | 0.5475 |
| Random graph true pairs | 15 | `conceptnet:card_games:1524` | card games | 淟猾 | けぉの | 0.4005 |
| Random graph true pairs | 16 | `conceptnet:lizard:1349` | lizard | 送乵 | ゑべおせ | 0.5613 |
| Random graph true pairs | 17 | `flour.n.01` | flour | 觡騳靱 | とえり | 0.5084 |
| Random graph true pairs | 18 | `factotum.n.01` | factotum | 鋈鼢 | ひむつ | 0.3704 |
| Random graph true pairs | 19 | `cairene.n.01` | cairene | 刴墯貽梷 | わ | 0.3701 |
| Random graph true pairs | 20 | `conceptnet:gentleman:1395` | gentleman | 魛慌 | でた | 0.4035 |
| Random graph true pairs | 21 | `workman.n.01` | workman | 龞賙貺 | ゎぁえ | 0.4457 |
| Random graph true pairs | 22 | `back.n.01` | back | 锏昞 | よす | 0.4486 |
| Random graph true pairs | 23 | `conceptnet:microbiology:1405` | microbiology | 厈 | むぇきつ | 0.5348 |
| Random graph true pairs | 24 | `salad.n.01` | salad | 犰蘟劋绎 | ちべとち | 0.5650 |
| Random graph true pairs | 25 | `church_tower.n.01` | church tower | 婴螵寵蠒 | はゅ | 0.3667 |
| Random graph true pairs | 26 | `princess.n.01` | princess | 蝤宧萑軜 | ぃでよ | 0.5524 |
| Random graph true pairs | 27 | `worker.n.01` | worker | 鑱溟溹 | ぅぅづ | 0.4401 |
| Random graph true pairs | 28 | `decoration.n.01` | decoration | 鬫稈跫 | ふがゆ | 0.6056 |
| Random graph true pairs | 29 | `part.n.03` | part | 闫犀鸆 | じどゆ | 0.5810 |
| Random graph true pairs | 30 | `conceptnet:museum:1573` | museum | 途鷈涐 | ぺけめひる | 0.5992 |
| Far at 100% | 1 | `bit.n.02` | bit | 顃幚萖 | ゕるけ | 1.0466 |
| Far at 100% | 2 | `insect.n.01` | insect | 疘膻佝 | ぜゃょず | 0.9511 |
| Far at 100% | 3 | `nervous_system.n.01` | nervous system | 跲嬾芖 | がづゕほ | 0.9508 |
| Far at 100% | 4 | `sail.n.01` | sail | 屪嬽謵諝 | は | 0.9375 |
| Far at 100% | 5 | `cap.n.04` | cap | 斒 | やいう | 0.9036 |
| Far at 100% | 6 | `conceptnet:closet:1314` | closet | 演兹鉯洕 | ゅだゑつ | 0.8980 |
| Far at 100% | 7 | `executive.n.01` | executive | 咶囹 | ねたぱ | 0.8902 |
| Far at 100% | 8 | `paint.n.01` | paint | 芔濣笚圗 | せせゎぅ | 0.8818 |
| Far at 100% | 9 | `conceptnet:slang:1201` | slang | 瑓荠岡 | そは | 0.8748 |
| Far at 100% | 10 | `officeholder.n.01` | officeholder | 獰癹夔雩 | ぜつど | 0.8625 |
| Far at 100% | 11 | `feather_star.n.01` | feather star | 耽疍築奊 | らづせみ | 0.8546 |
| Far at 100% | 12 | `official.n.01` | official | 拸 | けひご | 0.8522 |
| Far at 100% | 13 | `conceptnet:computing:1203` | computing | 庒闷鳐 | ごぶ | 0.8422 |
| Far at 100% | 14 | `hotel.n.01` | hotel | 獟樽乵 | うごぃ | 0.8421 |
| Far at 100% | 15 | `conceptnet:drawer:1703` | drawer | 覦 | ゆのぼれ | 0.8409 |
| Far at 100% | 16 | `conceptnet:bottom_of_sea:1537` | bottom of sea | 耒銋飲汘暗 | はむせ | 0.8340 |
| Far at 100% | 17 | `design.n.04` | design | 程焄唌玸 | す | 0.8301 |
| Far at 100% | 18 | `nerve_cell.n.01` | nerve cell | 蔎攱餰鸐 | ぱるらび | 0.8300 |
| Far at 100% | 19 | `conceptnet:box:1293` | box | 踁糍 | ぎゔ | 0.8287 |
| Far at 100% | 20 | `bite.n.04` | bite | 偌脪 | べぃ | 0.8207 |
| Far at 100% | 21 | `house.n.01` | house | 诏饶巑 | ぼよ | 0.8176 |
| Far at 100% | 22 | `administrator.n.01` | administrator | 麴 | ばゃゕげ | 0.8125 |
| Far at 100% | 23 | `choice_morsel.n.01` | choice morsel | 吰琡笇樷 | ぁるぇ | 0.8020 |
| Far at 100% | 24 | `conceptnet:pejorative:1485` | pejorative | 誠鎃紨夣 | ひ | 0.7999 |
| Far at 100% | 25 | `earflap.n.01` | earflap | 墸时倪 | ばえほ | 0.7997 |
| Far at 100% | 26 | `clerk.n.01` | clerk | 忌屼硰垱 | ぐぽせも | 0.7980 |
| Far at 100% | 27 | `defile.n.01` | defile | 迁朌蔭茼 | たでとれ | 0.7978 |
| Far at 100% | 28 | `address.n.02` | address | 蛹 | うそこむ | 0.7970 |
| Far at 100% | 29 | `necessity.n.02` | necessity | 譛綌黽靅 | わばこま | 0.7957 |
| Far at 100% | 30 | `patch.n.03` | patch | 祉乹囑綜唍 | げふくに | 0.7948 |

## Visualizations

- `visualizations/cosine_distance_trajectory.png`
- `visualizations/pca_selected_pairs.png`

Note: PCA panels are reduced separately per model, so they are useful for within-condition geometry, not absolute cross-condition axis comparison. The PCA figure labels representative points by original English synset lemma; artificial tokens remain in `pca_representative_pairs.csv`.
