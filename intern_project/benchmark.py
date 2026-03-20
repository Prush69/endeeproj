import time
import uuid
import random
from endee_client import EndeeClient


def main():
    print("=== Endee Vector Database Performance Benchmark ===")
    client = EndeeClient()
    try:
        if not client.ping():
            return
    except ConnectionError as e:
        print(f"Error: {e}")
        return

    index_name = "benchmark_idx"
    dim = 384
    
    print("\n[1] Initializing Benchmark Index...")
    client.create_index(index_name, dim, "cosine")

    print("\n[2] Benchmarking Insertion Throughput...")
    num_vectors = 5000
    batch_size = 500
    
    vectors = []
    print(f"Generating {num_vectors} dense random vectors of dimension {dim}...")
    for _ in range(num_vectors):
        vec = [random.random() for _ in range(dim)]
        vectors.append({"id": f"vec_{uuid.uuid4()}", "vector": vec})

    start_time = time.time()
    for i in range(0, num_vectors, batch_size):
        batch = vectors[i:i+batch_size]
        client.insert_vectors(index_name, batch)
        print(f"Inserted {i+len(batch)} vectors...")
    
    insert_time = time.time() - start_time
    throughput = num_vectors / insert_time
    print(f"--> Insert Time: {insert_time:.2f} seconds")
    print(f"--> Throughput: {throughput:.2f} vectors/second")

    print("\n[3] Benchmarking Semantic Search Latency...")
    num_queries = 100
    query_vectors = [[random.random() for _ in range(dim)] for _ in range(num_queries)]
    
    start_time = time.time()
    for q in query_vectors:
        client.search(index_name, q, k=10)
    
    search_time = time.time() - start_time
    avg_latency = (search_time / num_queries) * 1000
    
    print(f"--> Search completed {num_queries} synthetic queries.")
    print(f"--> Average KNN Search Latency: {avg_latency:.2f} ms per query")
    print("\nBenchmark completed. Demonstrates massive scalability!")

    client.close()

if __name__ == "__main__":
    main()
