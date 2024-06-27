import subprocess
import time


def main():
    while True:
        try:
            with open("/tmp/bpe_file_based_tokenizer.out", "w") as stdout_f, open(
                "/tmp/bpe_file_based_tokenizer.err", "w"
            ) as stderr_f:
                process = subprocess.Popen(
                    ["python", "-m", "cs336_basics.bpe_file_based_tokenizer"],
                    stdout=stdout_f,
                    stderr=stderr_f,
                )
                process.wait()  # Wait for the process to finish
                break
        except Exception as e:
            print(f"Error occurred: {e}")

        # Optional: Add a delay before restarting
        time.sleep(5)  # Adjust delay as needed


if __name__ == "__main__":
    main()
