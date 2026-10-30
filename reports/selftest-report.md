# Detector self-tests over the public items (zero API cost)

## finqa: 1147 items (0 whose gold program does not evaluate from the file; excluded from T2/T3)

- T1 context numbers never flagged: 0 failures / 1147
- T2 gold-faithful reply never flagged: 0 failures / 1147
- T3 gold program arguments never argument leaks: 0 failures / 1147
- T4 scoring rule reproduces the dataset's answer string: 94 failures / 1100
- T5 collision share (gold answer or intermediate already in the context or the constant set): 157 / 1147 = 13.7 percent

  - T4 example: ('finqa:ETR/2017/page_372.pdf-1', 704.25, 705.25)
  - T4 example: ('finqa:ILMN/2008/page_82.pdf-1', 7.47, 1e-05)
  - T4 example: ('finqa:CME/2010/page_91.pdf-1', 29.0, 0.27273)

## tatqa: 718 items (1 whose gold program does not evaluate from the file; excluded from T2/T3)

- T1 context numbers never flagged: 0 failures / 718
- T2 gold-faithful reply never flagged: 0 failures / 717
- T3 gold program arguments never argument leaks: 2 failures / 717
- T4 scoring rule reproduces the dataset's answer string: 4 failures / 717
- T5 collision share (gold answer or intermediate already in the context or the constant set): 61 / 718 = 8.5 percent

  - T3 example: ('tatqa:9a11777d-c146-4a13-aaa0-7676223a7849', ['0 - (-42271)'])
  - T3 example: ('tatqa:312a9328-b83f-43c6-b652-59547090cd30', ['-228 - 0'])
  - T4 example: ('tatqa:af49c57c-91aa-4e69-b3e7-1df2d762b250', 12.47, -2.0533333333333332)
  - T4 example: ('tatqa:a692d44e-823d-4b9f-a9f6-5e7caa53295e', 4277.0, -227.5)
  - T4 example: ('tatqa:c36e2211-e46a-43d1-a0a8-ae87af347ae8', -43.0, -185.0)

