from dataclasses import dataclass
from typing import List
from modifier import Modifier
from  feedback import Feedback


###Memory needs to store the full history across all iterations

@dataclass
class MemoryEntry: ##here stores single memory input
    iteration: int      # which loop iteration this belongs to
    modifier: Modifier  # the agent's proposed code modification for this iteration
    feedback: Feedback  # the environment's feedback for this iteration   
    

class Memory:
    def __init__(self):
        self.entries: List[MemoryEntry] = []  # list to store memory entries

    def add_entry(self, modifier: Modifier, feedback: Feedback) -> None:
        """Store a modifier+feedback pair after each iteration."""
        entry = MemoryEntry(iteration=modifier.iteration, 
                            modifier=modifier, 
                            feedback=feedback)
        self.entries.append(entry)

    def get_history(self) -> List[MemoryEntry]:
        """Return full history for the agent to reflect on."""
        return self.entries
    
    def get_latest_entry(self) -> MemoryEntry | None:
        """Return the most recent memory entry, or None if no entries exist."""
        if self.entries:
            return self.entries[-1]
        return None
    
    def get_entry_by_iteration(self, iteration: int) -> MemoryEntry | None:
        """Return the memory entry for a specific iteration, or None if not found."""
        for entry in self.entries:
            if entry.iteration == iteration:
                return entry
        return None
    
    def summary(self) -> str:
        """Generate a summary of the memory forthe LLM prompt debugging or analysis."""
        if not self.entries:
            return "No memory entries available."
        summary_lines = []
        for entry in self.entries:
            line = (f"Iteration {entry.iteration}: "
                    f"Modified Code: {entry.modifier.modified_code[:30]}..., "
                    f"rationale: {entry.modifier.rationale[:30]}..., "
                    f"Compiled: {entry.feedback.compiled}, "
                    f"Metrics: {entry.feedback.metrics_value}, "
                    f"Goal Met: {entry.feedback.goal_met}")
            summary_lines.append(line)
        return "\n".join(summary_lines)
    
    def clear(self) -> None:
        """Clear all memory entries."""
        self.entries.clear()