# Third-party code and licenses

We clone reference repos into `~/refs` to read READMEs, check versions/licenses, and borrow ideas
(or small snippets) where they fit. We do NOT vendor their code wholesale.

| Repo | URL | License | Use |
|------|-----|---------|-----|
| VectorizedMultiAgentSimulator (VMAS) | proroklab/VectorizedMultiAgentSimulator | **GPLv3** | Reference only (license is copyleft; we build our own simulator) |
| PettingZoo | Farama-Foundation/PettingZoo | MIT | API/pattern reference |
| hemac | ThalesGroup/hemac | Thales Non-Production License | Reference only (heterogeneous agent template) |
| IR2 | marmotlab/IR2 | Apache-2.0 | Ideas for GNN frontier exploration |
| m-explore-ros2 | robo-friends/m-explore-ros2 | BSD | Map merge + explore baseline (P5/P6) |

## Notes
- VMAS is GPLv3: we must NOT copy its source into our project unless we accept GPL. We build our
  own lightweight simulator instead, so no copyleft obligations attach to our code.
- hemac has a non-production license: reference only, no code copied.
- We will list additional deps (torch, torch_geometric, slam_toolbox, Nav2, RTAB-Map, etc.) and
  their licenses here as they are added.
