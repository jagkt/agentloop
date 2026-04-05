from dotenv import load_dotenv
load_dotenv()

import os
from agent import Agent
from environment import Environment
from memory import Memory
from providers.groq_provider import GroqProvider
from providers.ollama_provider import OllamaProvider
from providers.anthropic_provider import AnthropicProvider
from providers.openai_provider import OpenAIProvider



def get_provider(name: str) -> object:
    if name == "groq":
        return GroqProvider(
            api_key=os.environ["GROQ_API_KEY"]
        )
    elif name == "ollama":
        return OllamaProvider(model="llama3")
    elif name == "anthropic":
        return AnthropicProvider(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
    elif name == "openai":
        return OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"]
        )
    else:
        raise ValueError(f"Unknown provider: {name}")
    
    
def run_loop(task, original_code, goal_description, goal_metric, max_iterations, provider="ollama"):
    # 1. create environment
    environment = Environment(goal_metric=goal_metric)
    # 2. create memory
    memory = Memory()
    
    llm = get_provider(provider)
    
    # 3. create agent with everything injected
    agent = Agent(
        environment=environment,
        memory=memory,
        task=task,
        original_code=original_code,
        goal_description=goal_description,
        llm_provider=llm
    )
    # 4. run
    agent.run(max_iterations=max_iterations)

if __name__ == "__main__":
    run_loop(
        task="Optimize this CUDA matrix multiplication kernel for memory bandwidth",
        original_code="""
__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
""",
        goal_description="Achieve at least 95% memory bandwidth utilization",
        goal_metric=95.0,
        max_iterations=5,
        provider="groq"   # switch to "groq" anytime
    )
    