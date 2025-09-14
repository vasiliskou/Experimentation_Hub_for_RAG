import os
import time
import csv
from dotenv import load_dotenv

from generator import Generator

load_dotenv()

EXPERIMENTS_DIR = os.path.dirname(__file__)
OUTPUT_CSV = os.path.join(EXPERIMENTS_DIR, "generator_timings.csv")
print(f"Logging generator timings to: {OUTPUT_CSV}")

# Query for testing
QUERY = "Provide a 2-sentence summary of the European Union."

# Number of runs
RUNS = 3

# Generators to test
GENERATORS = [
    {"provider": "openai", "model_name": "gpt-5-mini"},
    {"provider": "openai", "model_name": "gpt-4o-mini"},
    {"provider": "openai", "model_name": "gpt-4o"},
    {"provider": "anthropic", "model_name": "claude-sonnet-4-20250514"},
    {"provider": "anthropic", "model_name": "claude-3-7-sonnet-20250219"},
    {"provider": "gemini", "model_name": "gemini-2.5-flash"},
    {"provider": "groq", "model_name": "llama-3.3-70b-versatile"},
    {"provider": "deepseek", "model_name": "deepseek-chat"},
]

def measure_generation(gen: Generator, query: str):
    start = time.time()
    _ = gen.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt=query
    )
    end = time.time()
    return round(end - start, 4)

def main():
    with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
        fieldnames = ["provider", "model", "run", "generation_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for g_cfg in GENERATORS:
            print(f"\nTesting generator: {g_cfg['provider']} - {g_cfg['model_name']}")
            gen = Generator(
                provider=g_cfg["provider"],
                model_name=g_cfg["model_name"],
                max_tokens=200
            )

            # Warm-up run
            print("Warm-up run to avoid cold start...")
            _ = measure_generation(gen, QUERY)

            # Timed runs
            for run in range(1, RUNS + 1):
                gen_time = measure_generation(gen, QUERY)
                print(f"{g_cfg['provider']} - {g_cfg['model_name']}, Run {run}, Time: {gen_time}s")
                writer.writerow({
                    "provider": g_cfg["provider"],
                    "model": g_cfg["model_name"],
                    "run": run,
                    "generation_time": gen_time
                })

if __name__ == "__main__":
    main()
