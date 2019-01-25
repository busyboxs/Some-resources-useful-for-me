# SSD 网络结构参数

|   层   |卷积核数量|kernel size|stride|padding|dilation|feature_scale|bbox number |
| :---:  |  :---: |   :---:   |:---: | :---: | :---:  |    :---:    |    :---:   |
|conv 1-1| 64     |3x3        |1     |       |        |300x300      |            |
|conv 1-2| 64     |3x3        |1     |       |        |             |            |
|conv 2-1| 128    |3x3        |1     |       |        |             |            |
|conv 2-2| 128    |3x3        |1     |       |        |150x150      |            |
|conv 3-1| 256    |3x3        |1     |       |        |             |            |
|conv 3-2| 256    |3x3        |1     |       |        |             |            |
|conv 3-3| 256    |3x3        |1     |       |        |75x75        |            |
|conv 4-1| 512    |3x3        |1     |       |        |             |            |
|conv 4-2| 512    |3x3        |1     |       |        |             |            |
|conv 4-3| 512    |3x3        |1     |       |        |38x38        |38x38x4=5776|
|conv 5-1| 512    |3x3        |1     |       |        |             |            |
|conv 5-2| 512    |3x3        |1     |       |        |             |            |
|conv 5-3| 512    |3x3        |1     |       |        |19x19        |            |
|   fc6  | 1024   |3x3        |1     |6      |6       |19x19        |            |
|   fc7  | 1024   |1x1        |1     |       |        |19x19        |19x19x6=2166|
|conv 6-1| 256    |1x1        |1     |       |        |             |            |
|conv 6-2| 512    |3x3        |2     |1      |        |10x10        |10x10x6=600 |
|conv 7-1| 128    |1x1        |1     |       |        |             |            |
|conv 7-2| 256    |3x3        |2     |1      |        |5x5          |5x5x6=150   |
|conv 8-1| 128    |1x1        |1     |       |        |             |            |
|conv 8-2| 256    |3x3        |1     |       |        |3x3          |3x3x4=36    |
|conv 9-1| 128    |1x1        |1     |       |        |             |            |
|conv 9-2| 256    |3x3        |1     |       |        |1x1          |1x1x4       |