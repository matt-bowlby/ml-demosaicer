import time

stack = []
results = []

def reset():
    stack = []
    results = []
    pass

def start(name: str = ""):
    stack.append((name, time.perf_counter()))

def stop():
    end = time.perf_counter()
    name, start = stack.pop()
    elapsed = end - start
    tab = len(stack)
    results.append(f"{"  " * tab}\"{name}\" Took {elapsed:.3f} seconds")
    print(f"\"{name}\" Took {elapsed:.3f} seconds")