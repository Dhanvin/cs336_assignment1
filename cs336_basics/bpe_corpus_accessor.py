
import regex as re
from typing import Dict, Tuple, List, Pattern, IO, Set
from collections import defaultdict
import pathlib
from .modifiable_priority_queue import ModifiablePriorityQueue, HeapItem

import cProfile
import pstats
import multiprocessing
import logging

from dataclasses import dataclass
import os

# Option 1: Very similar to current bce_tokenizer, with the exception that we store pre-tokens as keys directly
# Key difference: Do NOT store actual integers. Stick to byte-strings.
#
# Each pre-token is a tuple of byte-strings
# Pre-tokenization outputs a Dict {Tuple(bstring1, bstring2, ..): count}. This is "shelvable"
#       --> De-risk: Store all pre-tokens and check file-size
# For each Token-Pair in the Priority Queue, list the pre-tokens and Dict{Tuple(bstring1, bstring2, ..): index} where it is present. This is "shelvable"

# Option 2: Most data-access / searching directly on the file.
# > Pre-tokenize
# > Store pretoken-offsets {idx: file-offset} --> In RAM
# > For each token in vocab, store list of pretoken-idx, which will point to file-offset via  pretoken-offsets
# > Given any byte-string (token-pair or otherwise), we can extract each pretoken where it is present, and perform a search to find location, and find surrounding vocab tokens within the offsets of that token
#       -- O(n*m). Assuming that vocab bytestring is small, and pretokens aren't very large, this should be fine, esp. in comparison with finding the max element from priority queue.
#       -- Merge involves
# > At initialization, you can populate both vocab --> pretoken_idx store as well as pretoken_idx --> offset
# > At every merge, 

# Search for KNOWN BUGS:

logger = logging.getLogger(__name__)
def configure_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s - %(message)s')

def _char_to_byte_offset(unicode_string, char_pos):
    return len(unicode_string[:char_pos].encode('utf-8'))

@dataclass
class FileOffset:
    start: int
    end: int

def _find_bytestring(fp: IO, search_b: bytes, offset_span :FileOffset) -> List[int]:
    """
    Finds occurrances of |search_b| in |fp| within |offset_span|
    Return:
        A list of byte-offsets into fp
    """
    def find_all_indices(source: bytes, search: bytes):
        indices = []
        index = source.find(search)

        while index != -1:
            indices.append(index)
            index = source.find(search, index + 1)

        return indices
    # Save to revert
    original_offset = fp.tell()
    
    # Find occurrances
    fp.seek(offset_span.start)
    source_string = fp.read(offset_span.end - offset_span.start)
    found_indices = find_all_indices(source_string, search_b)
    
    # Revert file-pointer
    fp.seek(original_offset)
    
    # Correct for start of offset_span
    return list(map(lambda i: i + offset_span.start, found_indices))

# Used to parallelize the time-consuming operation of finding a bytestring in a file
def _find_token_pair_worker(file_path: str, token_pair_b: bytes, pretoken_id: int, 
                            file_offset_span: Tuple[int, int], invalid_locs: set) -> Tuple[int, List[int]]:
    with open(file_path, 'rb') as file:
        locations = _find_bytestring(file, token_pair_b, file_offset_span)
        filtered_locations = [l for l in locations if l not in invalid_locs and l + len(token_pair_b) not in invalid_locs]
    return pretoken_id, filtered_locations

class PretokenizedCorpusAccessor:
    def __init__(self, corpus_filepath, pattern: Pattern[str], chunk_size=1024):
        """
        Parse and split a large text file into chunks based on a regex pattern.
        
        Args:
            corpus_path (str): Path to the large text file.
            pattern (str): Regex pattern to split the file.
            chunk_size (int): Size of chunks to read from the file in bytes.
        
        Returns:
            A mapping from pretoken-idx -> File offset
        """
        self.corpus_filepath = corpus_filepath
        self.pretoken_offsets: Dict[int, FileOffset] = {}

        # Foe each pretoke, maintain all the file-offsets in between a 
        # previously merged token pair.
        # When finding token-pairs in the corpus, if any end of the token-pair
        # lies within this set, it is an invalid count
        self.pretoken_merged_offsets: Dict[int, Set[int]] = {}

        # For each byte-token-pair in vocabulary, a list of pretokens where the bytestring is present
        # Will be used to initialize heap and keep count
        self.token_pair_pretoken_map: Dict[bytes, Set[int]] = {}

        self.vocab_set = set([bytes([i]) for i in range(256)])
        self.merge_list: List[Tuple[bytes, bytes]] = []

        # Intitialize pattern-matching for unicode
        compiled_pattern = re.compile(pattern, re.UNICODE)
        try:
            # Open the corpus in text-mode assuming 
            tmp_token_pair_counter = {}
            with open(self.corpus_filepath, 'rb') as file:
                print(f"Reading {self.corpus_filepath}")
                buffer = b'' # Must be a string to perform pattern-matching
                buffer_start_offset = 0
                chunk_count = 0
                pretoken_count = 0
                eof_offset = os.path.getsize(self.corpus_filepath)

                while file.tell() < eof_offset:
                    # Read a chunk from the file
                    chunk = file.read(chunk_size)
                    chunk_count += 1

                    # # DEBUG: Use a smaller subset
                    # if chunk_count > 5000:
                    #     break

                    # Track % of file read
                    if chunk_count % 1024 == 0:    
                        percent_file_read = float(buffer_start_offset) * 100 / float(eof_offset)
                        logger.info(f'% file read: {percent_file_read: .2f}')

                    # Full chunk is past EOF resulting in empty string
                    if not chunk:
                        # Process any remaining buffer
                        chunk_count += 1

                        # Move the file pointer to the start of buffer and read until end of file
                        file.seek(buffer_start_offset)
                        chunk = file.read(eof_offset - buffer_start_offset)
                        assert chunk, "Failed to consume last part of file"

                    buffer += chunk

                    # Split buffer into matches after decoding to unicode characters
                    unicode_buffer = buffer.decode('utf-8', errors='ignore')
                    matches = list(compiled_pattern.finditer(unicode_buffer))
                    # breakpoint()
                    if matches:
                        # If there are matches, process them
                        last_match_end_byte_offset = 0
                        for match in matches:
                              
                            # Iterate through each byte in match.group()
                            pretoken_utf8_seq = match.group().encode('utf-8')
                            # Ignore pretokens that have only one since there it won't have any impact during training
                            if len(pretoken_utf8_seq) < 2:
                                continue
                            
                            self.pretoken_merged_offsets[pretoken_count] = set() # Initially no offset is within a merge                  
                            
                            # Add |buffer_start_offset| to get file-offset
                            # Note that match.end() would only include the number of unicode characters and NOT Utf-8 bytes
                            byte_offset_start = _char_to_byte_offset(unicode_buffer, match.start()) + buffer_start_offset
                            byte_offset_end = _char_to_byte_offset(unicode_buffer, match.end()) + buffer_start_offset
                            last_match_end_byte_offset = byte_offset_end
                            self.pretoken_offsets[pretoken_count] = FileOffset(start=byte_offset_start,
                                                                        end=byte_offset_end)
                            for idx in range(len(pretoken_utf8_seq) - 1):
                                token_byte_pair = bytes([pretoken_utf8_seq[idx], pretoken_utf8_seq[idx + 1]])
                                if token_byte_pair not in self.token_pair_pretoken_map:
                                    self.token_pair_pretoken_map[token_byte_pair] = set()
                                    tmp_token_pair_counter[(token_byte_pair[:1], token_byte_pair[1:2])] = 0
                                self.token_pair_pretoken_map[token_byte_pair].add(pretoken_count)
                                tmp_token_pair_counter[(token_byte_pair[:1], token_byte_pair[1:2])] += 1

                            pretoken_count += 1
                        # Keep the part of the buffer after the last match
                        buffer = buffer[last_match_end_byte_offset - buffer_start_offset:]
                        buffer_start_offset = last_match_end_byte_offset
            print("Corpus Pretokenized!")
            self.priority_queue = ModifiablePriorityQueue.heapify([
                HeapItem(b''.join(token_pair_b), (count, token_pair_b))
                    for token_pair_b, count in tmp_token_pair_counter.items()
            ])
            print("Heap initialized!")
                    
        except FileNotFoundError:
            print(f"The file {self.corpus_filepath} does not exist.")
  
    def _find_token_pair_parallel(self, token_pair_b: bytes) -> Dict[int, List[int]]:

        pretoken_ids = list(self.token_pair_pretoken_map[token_pair_b])
        file_offsets = list(map(self.pretoken_offsets.get, pretoken_ids))
        
        logger.debug(f"> Searching in {len(file_offsets)} locations")
        
        # Prepare arguments for multiprocessing, parallelizing based on pre-token and corresponding offsets
        worker_args = [(self.corpus_filepath, token_pair_b, pretoken_id, file_offset_span, self.pretoken_merged_offsets[pretoken_id])
                    for pretoken_id, file_offset_span in zip(pretoken_ids, file_offsets)]
        
        # Use multiprocessing to parallelize the search
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            # Apply worker function to all the prepared arguments in parallel.
            results = pool.starmap(_find_token_pair_worker, worker_args)
        
        # Combine results
        out = {pretoken_id: locations for pretoken_id, locations in results}
        total_cnt = sum(len(locations) for locations in out.values())
        
        logger.debug(f"> Found in {total_cnt} locations")
        return out

    def merge(self, merge_pair: Tuple[bytes, bytes], expected_freq: int=None) -> Dict[Tuple[bytes, bytes], int]:
        """
        Updates self.vocab and self.pretoken_merged_offsets 
        Only update self.token_pair_pretoken_map for new tokens, but not old ones
        Returns:
            Changed counts for each token-pair
        """
        def introduce_token_pair(new_tp: bytes, pretoken_idx: int):
            if new_tp not in self.token_pair_pretoken_map:
                self.token_pair_pretoken_map[new_tp] = set()            
            self.token_pair_pretoken_map[new_tp].add(pretoken_idx)

        self.merge_list.append(merge_pair)
        changed_counts = defaultdict(int)
        token_pair_b = merge_pair[0] + merge_pair[1]
        token_pair_len = len(token_pair_b)        
    
        token_pair_file_offsets = self._find_token_pair_parallel(token_pair_b)
        noccurrances = sum([len(locations) for locations in token_pair_file_offsets.values()])

        # TODO: KNOWN BUGS: Uncomment to trigger
        # if expected_freq is not None and expected_freq != noccurrances:
        #     breakpoint()

        logger.debug(f"> Found in {noccurrances} / {expected_freq}  locations")

        # Important: Add to vocab *before* merging. Handle edge case if maximal token is same as current token.
        #
        # E.g. if (b'e', b'd') are merged and we have multiple occurrances of "needed",
        # We should be updating the priorities of token-pairs b'de'.
        # However, if we have duplicate occurrances, the new token is no longer b'ded' but b'eded'.
        # So, to remove an element, we must ignore the added token, but while adding the elemnt we must include it.
        #
        # Therefore, we first add to vocabulary so that b'eded' is returned when |_find_maximal_token_before| is called
        # Then if the maximal returned token is same as the current token, then the pre-merged token is used to update the priority queue
        self.vocab_set.add(token_pair_b)
        with open(self.corpus_filepath, 'rb') as file:
            for pretoken_idx, offsets in token_pair_file_offsets.items():
                pretoken_offset_span = self.pretoken_offsets[pretoken_idx]

                file.seek(pretoken_offset_span.start-1)
                pretoken = file.read(pretoken_offset_span.end - pretoken_offset_span.start)

                # NOTE: Mark offsets as merged
                for offset in offsets:
                    # logger.debug(f"Processing {token_pair_b} for {pretoken} at {offset}")
                    # Mark all offsets that fall *within* |token_pair_b|.
                    self.pretoken_merged_offsets[pretoken_idx].update(list(range(offset + 1, offset + token_pair_len))) 
                    # Look behind from beginning for pretoken
                    file.seek(offset)
                    token_before = self._find_maximal_token_before(file, pretoken_offset_span)
                    if token_before:
                        # Handle edge case that maximal token is same as current token
                        old_token_pair_before = (merge_pair[1], merge_pair[0]) if token_before == token_pair_b else (token_before, merge_pair[0])
                        merged_token_pair_before = (token_before, token_pair_b)                  
                        # Account for new entries and maintain a fresh running count
                        introduce_token_pair(b''.join(merged_token_pair_before), pretoken_idx)
                        changed_counts[merged_token_pair_before] += 1
                        changed_counts[old_token_pair_before] -= 1

                    # Look ahead until end of pre-token
                    file.seek(offset + token_pair_len)
                    token_after = self._find_maximal_token_after(file, pretoken_offset_span)
                    if token_after:
                        # Handle edge case that maximal token is same as current token
                        old_token_pair_after = (merge_pair[1], merge_pair[0]) if token_after == token_pair_b else (merge_pair[1], token_after)
                        merged_token_pair_after = (token_pair_b, token_after)
                        # Account for new entries and maintain a fresh running count
                        introduce_token_pair(b''.join(merged_token_pair_after), pretoken_idx)
                        changed_counts[merged_token_pair_after] += 1
                        changed_counts[old_token_pair_after] -= 1
            
        return changed_counts

    def train_one_step(self):
        freq, max_freq_token_pair = self.priority_queue.pop_task().priority
        logger.info(f"Merging {max_freq_token_pair} with freq: {freq}")
        changed_counts = self.merge(max_freq_token_pair, freq)

        # if max_freq_token_pair in set([(b' t', b'o'), (b'r', b'e')]):
        #     print("Changing to DEBUG mode")
        #     logger.setLevel(logging.DEBUG)
        # else:
        #     logger.setLevel(logging.INFO)
        
        for token_pair, count_change in changed_counts.items():
            token_pair_b = b''.join(token_pair)
            if self.priority_queue.contains(token_pair_b):
                count = self.priority_queue.get_priority(token_pair_b)[0]
                logger.debug(f'>> Priority of {token_pair_b} changed from {count} to {count + count_change}')
                count += count_change
                self.priority_queue.change_priority(
                    token_pair_b,
                    (count, token_pair),
                )
            else:
                logger.debug(f'>> Heap insertion: Adding {token_pair_b} with count {count_change}')
                # TODO: KNOWN BUGS: Uncomment to trigger
                # assert count_change > 0, f"Heap insertion error: Trying to add new token-pair {token_pair} with count {count_change}"
                self.priority_queue.add_task(token_pair_b, (count_change, token_pair))

    
    def _find_maximal_token_before(self, fp: IO, offset_span: FileOffset) -> bytes:
        """
        Return:
            The largest continuous byte-string from file |fp| in |self.vocab| before |fp.tell()| within |offset_span|
            Returns an empty string if not found
        """
        
        out = b''
        this_offset = fp.tell()
        assert this_offset >= offset_span.start and this_offset <= offset_span.end
        
        # Number of bytes to look behind
        lookbehind = 1
        while True:
            if this_offset - lookbehind < offset_span.start:
                break
            # Reset and byte-string
            fp.seek(this_offset - lookbehind)
            prev_bytes = fp.read(lookbehind)
            if prev_bytes in self.vocab_set:
                out = prev_bytes
                lookbehind += 1
            else:
                break # We are done looking

        fp.seek(this_offset)
        return out

    def _find_maximal_token_after(self, fp: IO, offset_span: FileOffset) -> bytes:
        """
        Return:
            The largest continuous byte-string from file |fp| in |self.vocab| after |fp.tell()| within |offset_span|
            Returns an empty string if not found
        """
        out = b''
        this_offset = fp.tell()
        assert this_offset >= offset_span.start and this_offset <= offset_span.end
        
        # Number of bytes to look ahead
        lookahead = 1
        while True:
            if this_offset + lookahead > offset_span.end:
                break
            # Get byte-string
            fp.seek(this_offset)
            next_bytes = fp.read(lookahead)
            if next_bytes in self.vocab_set:
                out = next_bytes
                lookahead += 1
            else:
                break # We are done looking

        fp.seek(this_offset)
        return out
    
    def print_pretoken_at(self, fp: IO, pretoken_idx: List[int]):
        restore_to = fp.tell()
        for idx in pretoken_idx:
            offset_span = self.pretoken_offsets[idx]
            fp.seek(offset_span.start)
            pretoken = fp.read(offset_span.end - offset_span.start)
            logger.debug(f"{idx}/{len(pretoken_idx)}: {pretoken.decode('utf-8')}")
            
        fp.seek(restore_to)
# Example usage

# Matches the following patterns:
# Contractions ('s, 'd, 'm, 't, ll, ve, re).
# Words composed entirely of letters (\p{L}).
# Words composed entirely of digits (\p{N}).
# Words composed of characters that are not whitespace, letters, or digits.
# Sequences of whitespace characters.
PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# python -m cs336_basics.bpe_corpus_accessor
if __name__ == "__main__":
    # DATASET_PATH = (pathlib.Path(__file__).resolve()).parent.parent
    # DATASET_FILE = 'tokenizer_mini_test'

    # DATASET_PATH = (pathlib.Path(__file__).resolve()).parent.parent / 'tests' / 'fixtures'
    # DATASET_FILE = 'corpus'
    import argparse
    import timeit
    import pickle


    parser = argparse.ArgumentParser(description="Example script with debug logging")
    parser.add_argument('--debug', action='store_true', help="Enable debug messages")
    args = parser.parse_args()
    configure_logging(args.debug)

    DATASET_PATH = (pathlib.Path(__file__).resolve()).parent.parent / 'data'
    DATASET_FILE = 'TinyStoriesV2-GPT4-train.txt'
    corpus_path = DATASET_PATH / DATASET_FILE
    # Get stored initialization locaiton
    stem = corpus_path.stem
    pickle_file_path = corpus_path.with_name(f"{stem}_init.pkl")
    
    start_time = timeit.default_timer()
    if pickle_file_path.exists():
        logger.info(f"Loading from stored state at {pickle_file_path}")
        with open(pickle_file_path, 'rb') as file:
            corpus_accessor = pickle.load(file)
    else:
        corpus_accessor = PretokenizedCorpusAccessor(DATASET_PATH / DATASET_FILE, PRETOKEN_PATTERN)
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(corpus_accessor, file)
    end_time = timeit.default_timer()
    print(f"Initialization time: {end_time - start_time: .2f} seconds")
    
    # DEBUG: Print pretokens
    # with open(corpus_accessor.corpus_filepath, 'rb') as file:
    #     corpus_accessor.print_pretoken_at(file, range(len(corpus_accessor.pretoken_offsets.keys())))
    # breakpoint()

    profiler = cProfile.Profile()
    profiler.enable()
    for i in range(100):
        start_time = timeit.default_timer()
        corpus_accessor.train_one_step()
        end_time = timeit.default_timer()
        print(f"Train step #{i} time: {end_time - start_time: .2f} seconds")
    profiler.disable()


    # Print profiling results
    stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumulative')
    stats.print_stats()

    # Note: Saving the Heap and corpus_accessor after a few merges should retain full state





