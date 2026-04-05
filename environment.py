from feedback import Feedback
from modifier import Modifier


class Environment:
    def __init__(self, goal_metric: float):
        self.goal_metric = goal_metric # e.g. 80.0 for 80% bandwidth utilization
        
    def _compile(self, code: str) -> tuple[bool, str | None]:
        """
        Compile the kernel code.
        Returns (success, error_message).
        Mock for now — replace with real nvcc call in KernelScope.
        """
        # TODO: subprocess.run(["nvcc", ...])
        print("[Environment] Compiling code...")
        if "error" in code.lower():
            return False, "Compilation failed: syntax error detected"
        return True, None
    
    def _profile(self, code: str) -> tuple[bool, str | None, float | None]:
        """
        Profile the compiled kernel.
        Returns (raw_output, metric_value).
        Mock for now — replace with real ncu call in KernelScope.
        """
        # TODO: subprocess.run(["ncu", ...])
        print("[Environment] Profiling code...")
        import random
        metric = round(random.uniform(49.0, 95.0), 2)  # mock metric
        raw_output = f"Memory bandwidth utilization: {metric}%"
        return raw_output, metric
    
    def run(self, modifier: Modifier) -> Feedback:
        """
        Main entry point. Receives a Modifier, compiles and profiles it,
        returns a Feedback object.
        """
        # Step 1: compile
        compiled, compile_error = self._compile(modifier.modified_code)
        
        # Step 2: profile only if compilation succeeded
        if not compiled:
            return Feedback(iteration=modifier.iteration, 
                            compiled=False, 
                            compiled_error=compile_error, 
                            profile_output=None, 
                            metrics_value=None, 
                            goal_met=False)
        
        # Step 3: profile
        profile_output, metric_value = self._profile(modifier.modified_code)
        
        # Step 4: check if goal is met
        goal_met = metric_value is not None and metric_value >= self.goal_metric
        
        return Feedback(iteration=modifier.iteration, 
                        compiled=True, 
                        compiled_error=None, 
                        profile_output=profile_output, 
                        metrics_value=metric_value, 
                        goal_met=goal_met)