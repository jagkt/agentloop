from dataclasses import dataclass

@dataclass
class Modifier:
    original_code: str  # the kernel code before change
    modified_code: str  # the agent's proposed new version
    rationale: str      # why the agent made this change
    iteration: int      # which loop iteration this belongs to
    