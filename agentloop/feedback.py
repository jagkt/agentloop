from dataclasses import dataclass
from typing import Optional

##agent reasons about the intent, while the environment report facts. they never overlap
@dataclass
class Feedback:
    iteration: int      # which loop iteration this belongs to
    compiled: bool     # whether the modified code compiled successfully
    compiled_error: Optional[str]  # error message if compilation failed
    profile_output: Optional[str]  # output from profiling the modified code
    metrics_value: Optional[float]  # the value of the performance metric after modification
    goal_met: bool       # whether the performance goal was met after modification
    