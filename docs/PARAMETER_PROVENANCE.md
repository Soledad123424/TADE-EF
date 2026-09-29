# Parameter provenance

The manuscript defines the equations but does not publish every numerical
parameter. This file prevents implementation choices from being presented as
paper-specified values.

| Parameter | Value in `paper.yaml` | Provenance |
|---|---:|---|
| event window | 20 ms | submitted manuscript, Sec. IV-A1 |
| frequency range/step | 20-1200 Hz / 20 Hz | archived experiment configuration |
| history/short windows | 1000/100 ms | archived experiment configuration |
| harmonics | 4 | archived experiment configuration |
| association distance | 60 px | archived experiment configuration |
| minimum segment / maximum segment | 300/1000 ms | archived experiment configuration |
| evidence decay | 0.95 | archived experiment configuration |
| UAV evidence threshold | 0.42 | submitted manuscript, Sec. IV-A3 |
| IoU/distance weights | 0.5/0.5 | explicit implementation choice; manuscript omits values |
| association maximum cost | 0.9 | explicit implementation choice; manuscript omits value |
| classification threshold | 0.5 | neutral implementation choice; distinct from evidence threshold |
| Non-UAV evidence threshold | -1.0 | archived experiment configuration |
| Non-UAV lock duration | 1000 ms | archived maximum segment duration |
| Huber epsilon | 1.35 | scikit-learn default; manuscript omits value |

Any tuned replacement must create a new configuration file and must not
silently overwrite `paper.yaml`.

