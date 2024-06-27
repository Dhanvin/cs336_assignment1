import time
import shelve

# Summary of learnings:
#   - Avoid re-opening file (strongly prefer batch op)
#   - Fairly efficient for few accesses. (seems to scale linearly in number of reads / writes)

# Number of elements to test
N = 100000

# Generate some data
data = {f"key_{i}": i for i in range(N)}

# Test in-RAM dictionary
start_time = time.time()
ram_dict = {}
for k, v in data.items():
    ram_dict[k] = v
ram_write_time = time.time() - start_time

start_time = time.time()
for k in data:
    _ = ram_dict[k]
ram_read_time = time.time() - start_time
print(f"In-RAM dictionary write time: {ram_write_time:.3f} seconds")
print(f"In-RAM dictionary read time: {ram_read_time:.3f} seconds")

# Test shelve: Batch read / write: 30x slower than RAM
start_time = time.time()
with shelve.open("/tmp/test_shelve.db") as shelf:
    for k, v in data.items():
        shelf[k] = v
shelve_write_time_batch = time.time() - start_time

start_time = time.time()
with shelve.open("/tmp/test_shelve.db") as shelf:
    for k in data:
        _ = shelf[k]
shelve_read_time_batch = time.time() - start_time

print(f"Batch Shelve write time: {shelve_write_time_batch:.2f} seconds")
print(f"Batch Shelve read time: {shelve_read_time_batch:.2f} seconds")

# Test shelve: Batch read / write scales linearly with number of reads
start_time = time.time()
max_cnt = 10000
with shelve.open("/tmp/test_shelve.db") as shelf:
    cnt = 1
    for k, v in data.items():
        shelf[k] = v
        cnt += 1
        if cnt == max_cnt:
            break

shelve_write_time_batch = time.time() - start_time

start_time = time.time()
with shelve.open("/tmp/test_shelve.db") as shelf:
    cnt = 1
    for k in data:
        _ = shelf[k]
        cnt += 1
        if cnt == max_cnt:
            break
shelve_read_time_batch = time.time() - start_time

print(f"{max_cnt} shelve write time: {shelve_write_time_batch:.2f} seconds")
print(f"{max_cnt} shelve read time: {shelve_read_time_batch:.2f} seconds")

# # Test shelve: Iterative read / write: 20-30x slower than Batch
# start_time = time.time()
# for k, v in data.items():
#     with shelve.open('/tmp/test_shelve.db') as shelf:
#         shelf[k] = v
# shelve_write_time_iter = time.time() - start_time

# start_time = time.time()
# for k in data:
#     with shelve.open('/tmp/test_shelve.db') as shelf:
#         _ = shelf[k]
# shelve_read_time_iter = time.time() - start_time

# print(f'Iterative Shelve write time: {shelve_write_time_iter:.2f} seconds')
# print(f'Iterative Shelve read time: {shelve_read_time_iter:.2f} seconds')
