# GPU Memory Usage

## H200s

All at 2k max context length.

| Model Size | Batch Size (BS) | Memory Usage | Status    |
|:-----------|:---------------:|:------------:|:----------|
| **34M**    |       72        |    115 GB    | ✅ Tested  |
| **48M**    |       72        |    121 GB    | ✅ Tested  |
| **63M**    |       72        |    121 GB    | ✅ Tested  |
| **93M**    |       64        |    117 GB    | ✅ Tested  |
| **153M**   |       64        |    131 GB    | ✅ Tested  |
| **344M**   |       48        |     OOM      | ⏳ Pending |
| **344M**   |       46        |    130 GB    | ⏳ Pending |
| **499M**   |       40        |     OOM      | ⏳ Pending |
| **499M**   |       38        |    143 GB    | ⏳ Pending |
| **660M**   |       64        |     TBD      | ⏳ Pending |
| **806M**   |       64        |     TBD      | ⏳ Pending |
