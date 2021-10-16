Here is the diagnostic tool for BMVC 2021 paper **Diagnosing Errors in Video Relation Detectors**.

We provide a tiny ground truth file `demo_gt.json`, and tiny video relation detection results (under `demo_det/`) for the demo. 

Run `vrd_analysis.py` will get false positive analysis figure, false negative analysis figure, mAP gain analysis and relation characteristics analysis figures. These figures are saved as PDF files corresponding.

If you find this diagnostic tool useful in your research please cite:
```
@inproceedings{chen2021diagnosing,
  title={Diagnosing Errors in Video Relation Detectors},
  author={Chen, Shuo and Pascal, Mettes and Snoek, Cees GM},
  booktitle={BMVC},
  year={2021}
}
