# Canonical source reconstruction

Reconstructed from verified private originals corresponding to pinned November source captures. The audit commits were not merged into this branch. The private operator mapping records original paths/bytes, transformed public snapshots and canonical destination/hash; public snapshot redactions are not runtime configuration.

| Canonical destination | Source SHA-256 | Captured release |
|---|---|---|
| `app.py` | `969162565960957cd7cbd7bd3dc44d0a5fa175ad9d3a5c09ed5b82afe1359f2b` | WebAuth wake-songservice |
| `gpu_runner.py` | `6d7866fd74668d65934f4f1d9d312ccc5a71ac87d99a572e63965e59fa2dc792` | WebAuth wake-songservice |
| `compatibility/client-sdk/aihub_gpu_runner/client.py` | `b805c95713e6c78bede1b7df913e37585fe54f41d0c703b2ff67afbd9d41eab9` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/config.py` | `a4f46bcd5dea07e38329f340cc2c5bbb2664052a0f1ffbb64d0c212eab0b9384` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/dual_gpu.py` | `e0049245e4c30e10223a824cf9003177fda855559cda6af380a12b018e707ee8` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/backends.py` | `1abf39f92b9d384eeba67290b595dda81ef9f8f813c939b6e07b3999975ed142` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/host_stage.py` | `bfbd202fba265953ce1b96ff0a1fe850779456fa3a6752f4414c2532c9814f13` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/api.py` | `51276ce7ca420d3d3bc7865ffb3935201ee2d787d0f02797c115f07dbd1b23cb` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/scope_transition.py` | `c8307970f14f0fb77d82a541ba6ec2e3117283c1fc7e9cc75e6fd3b32c5fcc99` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/core.py` | `defe249d9347f3ccdd84e91760a6d1e2e635faac345986b748115cd9f5779ce8` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/__init__.py` | `658351de765a1117a788de44456c1a96a64eb12e5313e711fdb18046ada034fe` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/__main__.py` | `c52f38be3175058f8121029fa022776e4815c63d1148fe286a4fbc87aa481705` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/music_cleanup.py` | `9bd57e32d55fa6b192a86605d7ea32ca195b25ffea05133f6249529172f4bbe1` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/recovery.py` | `bdf2eac516560930463a30143caeaf639e5c5d03d5eec9e50d3b835a273185a2` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/lock_migration.py` | `d05fd3befffc87882c77179d64a81bb9064f59f84ca551784a7feb440ff687be` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/http_deadline.py` | `a7066fe40e67872e793bc0ef8e268931dca94e626775105af0c0bd881717265f` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/ollama_cleanup.py` | `974e7520d28ef544c50e3c196f50c409f0d8912aa5410f10f2c16707a83a319e` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/wallpaper_lifecycle.py` | `dd7565be27d52690344d664f32915b9b15fbf0b84e4f0318cfef910a2ce5404f` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/recovery_summary.py` | `7a9110967ea790899a0402473ce6a1f6141f96e11905633bdae4676f0155ad27` | WebAuth installed SDK |
| `compatibility/client-sdk/aihub_gpu_runner/reboot_contract.py` | `843529b9fb5f208ee81a89ee44a3cca66eb2946adcf2ccdcadacee0b96fcc805` | WebAuth installed SDK |

Source hashes describe the reconstruction baseline, before explicitly reviewed behavior changes. Current Git commits/build manifests identify candidate bytes. No production deployment is implied.

Canonical plan: [runner roadmap](https://github.com/bojinda/aihub-gpu-runner/blob/codex/production-reliability-candidate-20261111/roadmap.md).
