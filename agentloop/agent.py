#from anthropic import Anthropic
import re
from agentloop.environment import Environment
from agentloop.memory import Memory
from agentloop.modifier import Modifier
from agentloop.providers.base import BaseLLMProvider

class Agent:
    def __init__(
        self, 
        environment: Environment, 
        memory: Memory,
        task: str,
        original_code: str,
        goal_description: str,
        llm_provider: BaseLLMProvider
        ):
        
        self.environment = environment
        self.memory = memory
        self.task = task
        self.original_code = original_code
        self.goal_description = goal_description
        self.llm = llm_provider
        # self.client = Anthropic() # Initialize Anthropic client

    #think builds the prompt using memory.summary() — the LLM sees the full history
    def think(self, iteration: int) -> str:
        """
        Call Claude with the task, original code, goal, and memory history.
        Returns the LLM's reasoning as a string.
        """
#         prompt = f"""
# You are an expert GPU kernel optimization agent.

# ## Task
# {self.task}

# ## Goal
# {self.goal_description}

# ## Original Code
# {self.original_code}

# ## History of Previous Attempts
# {self.memory.summary()}

# ## Instructions
# Analyze the history above. Propose a specific code modification that will 
# improve performance toward the goal. Explain your reasoning clearly.
# Then return the modified code.

# Respond in this exact format:
# RATIONALE: <your reasoning>
# CODE:
# <modified code here>
# """
        prompt = f"""
        ## Instructions
Analyze the history above. Propose a specific code modification that will
improve performance toward the goal. Explain your reasoning clearly.

Return a COMPLETE, COMPILABLE CUDA program including:
- All necessary #include statements
- The kernel function
- A main() function that:
  - Allocates matrices A, B, C on the GPU using cudaMalloc (use N=512)
  - Initializes data
  - Launches the kernel with appropriate grid and block dimensions
  - Calls cudaDeviceSynchronize()
  - Frees memory with cudaFree

# IMPORTANT CUDA RULES:
# - Shared memory arrays must use compile-time constants for dimensions, not variables
# - Use #define TILE_SIZE 16 at the top and reference TILE_SIZE in shared memory declarations
# - Never use a runtime variable as a shared memory array dimension
# - If this is the first iteration with no history, start with a correct tiled matmul implementation using TILE_SIZE=16.

IMPORTANT CUDA RULES:
- Shared memory arrays must use compile-time constants for dimensions
- Use #define TILE_SIZE 16 at the top
- Never use runtime variables as shared memory dimensions
- Always use cudaDeviceSynchronize() — never truncate it
- Always write complete, valid C++ — never leave statements incomplete
- The program must compile with: nvcc program.cu -o program

The program must compile with: nvcc program.cu -o program

Respond in this exact format:
RATIONALE: <your reasoning>
CODE:
<complete CUDA program here>
        """
        print(f"[Agent] Thinking about iteration {iteration}...")
        return self.llm.generate_response(prompt)
        
        ##for anthropic
        # message = self.client.messages.create(
        #     model="claude-opus-4-5",
        #     max_tokens=1024,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return message.content[0].text
    
    
    # def _strip_code_fences(self, code: str) -> str:
    #     """Remove markdown code fences and any trailing text from LLM response."""
    #     code = re.sub(r"```[\w]*\n", "", code)  # remove opening fence
    #     code = re.sub(r"```.*", "", code, flags=re.DOTALL)  # remove closing fence and everything after
    #     result = code.strip()
    #     print(f"[Agent] Newlines after strip: {result.count(chr(10))}")
    #     return result
    
    def _strip_code_fences(self, code: str) -> str:
        # extract only what's between the code fences
        match = re.search(r"```[\w]*\n(.*?)```", code, re.DOTALL)
        if match:
            return match.group(1).strip()
        # fallback - remove any remaining fences
        code = re.sub(r"```[\w]*\n?", "", code)
        code = re.sub(r"```", "", code)
        return code.strip()
    
    
    #act calls think and parses the response into a clean Modifier
    def act(self, iteration: int) -> Modifier:
        """
        Call think() to get LLM response, parse it into a Modifier object.
        """
        response = self.think(iteration)
        # print(f"[DEBUG] Raw response:\n{response}\n")
        
        # Parse rationale and code from response
        rationale = ""
        modified_code = ""
        
        # if "RATIONALE:" in response and "CODE:" in response:
        #     parts = response.split("CODE:")
        #     rationale = parts[0].replace("RATIONALE:", "").strip()
        #     modified_code = parts[1].strip()
        #     modified_code = modified_code.replace("```cuda", "").replace("```python", "").replace("```c", "").replace("```", "").strip()
        if "RATIONALE:" in response and "CODE:" in response:
            parts = response.split("CODE:")
            rationale = parts[0].replace("RATIONALE:", "").strip()
            modified_code = self._strip_code_fences(parts[1])
            # print(f"[DEBUG] Stripped code:\n{modified_code}\n")

        else:
            # Fallback if LLM doesn't follow format exactly
            rationale = "No rationale provided"
            #modified_code = response.strip()
            modified_code = self._strip_code_fences(response)
        
        return Modifier(
            original_code=self.original_code,
            modified_code=modified_code,
            rationale=rationale,
            iteration=iteration
        )
    
    #reflect just reads memory — no LLM call needed, pure logic
    def reflect(self) -> bool:
        """
        Read the latest feedback from memory.
        Returns True if the agent should continue, False if goal is met.
        """
        last = self.memory.get_latest_entry()
        if last is None:
            return True  # no iterations yet, keep going

        if last.feedback.goal_met:
            print(f"[Agent] Goal met at iteration {last.iteration} "
                  f"with metric={last.feedback.metrics_value}. Stopping.")
            return False

        print(f"[Agent] Reflecting... metric={last.feedback.metrics_value}, "
              f"goal not yet met. Continuing.")
        return True

    #run orchestrates everything with a clean early exit
    def run(self, max_iterations: int) -> None:
        """
        Main agent loop.
        Runs until goal is met or max_iterations is reached.
        """
        print(f"[Agent] Starting loop. Max iterations: {max_iterations}")
        for i in range(max_iterations):
            print(f"\n[Agent] ---- Iteration {i + 1} ----")

            # Think + Act → produce a modifier
            modifier = self.act(iteration=i + 1)
            print(f"[Agent] Rationale: {modifier.rationale}")

            # Environment compiles + profiles → produce feedback
            feedback = self.environment.run(modifier)
            print(f"[Agent] Feedback: compiled={feedback.compiled}, "
                  f"metric={feedback.metrics_value}, goal_met={feedback.goal_met}")

            # Store in memory
            self.memory.add_entry(modifier, feedback)

            # Reflect — should we continue?
            if not self.reflect():
                break

        print("\n[Agent] Loop complete.")