from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools

import numpy as np
import tqdm

from typing import Iterable, Dict
import gzip
import json
import os
import sys
import re

# ==========
# data
ROOT = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL = os.path.join(ROOT,  "datasets", "HumanEval.jsonl.gz")


def read_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))


# ==========
# execution
import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from typing import Dict, Optional


def unsafe_execute(problem: Dict, completion: str, timeout: float, result):
    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        unlink = os.unlink
        reliability_guard()

        # Construct the check program and run it.
        check_program = (
            problem["prompt"]
            + completion
            + "\n"
            + problem["test"]
            + "\n"
            + f"check({problem['entry_point']})"
        )

        

        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    # WARNING
                    # This program exists to execute untrusted model-generated code. Although
                    # it is highly unlikely that model-generated code will do something overtly
                    # malicious in response to this test suite, model-generated code may act
                    # destructively due to a lack of model capability or alignment.
                    # Users are strongly encouraged to sandbox this evaluation suite so that it
                    # does not perform destructive actions on their host or network. For more
                    # information on how OpenAI sandboxes its code, see the accompanying paper.
                    # Once you have read this disclaimer and taken appropriate precautions,
                    # uncomment the following line and proceed at your own risk:
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

        os.unlink = unlink


def check_correctness(
    problem: Dict, completion: str, timeout: float, completion_id: Optional[int] = None
) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(problem, completion, timeout, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    import builtins
    builtins.help = None 
    # __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None

# ==========
# evaluation

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


# def evaluate_functional_correctness(
#     sample_file: str,
#     k: List[int] = [1, 10, 100],
#     n_workers: int = 4,
#     timeout: float = 3.0,
#     problem_file: str = HUMAN_EVAL,
# ):
#     """
#     Evaluates the functional correctness of generated samples, and writes
#     results to f"{sample_file}_results.jsonl.gz"
#     """

#     problems = read_problems(problem_file)
#     # import pdb;pdb.set_trace()

#     # Check the generated samples against test suites.
#     with ThreadPoolExecutor(max_workers=n_workers) as executor:

#         futures = []
#         completion_id = Counter()
#         n_samples = 0
#         results = defaultdict(list)

#         print("Reading samples...")
#         for sample in tqdm.tqdm(stream_jsonl(sample_file)):
#             task_id = sample["task_id"]
#             completion = sample["completion"]
#             args = (problems[task_id], completion, timeout, completion_id[task_id])
#             future = executor.submit(check_correctness, *args)
#             futures.append(future)
#             completion_id[task_id] += 1
#             n_samples += 1

#         assert len(completion_id) == len(problems), "Some problems are not attempted."

#         print("Running test suites...")
#         for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
#             result = future.result()
#             results[result["task_id"]].append((result["completion_id"], result))

#     # Calculate pass@k.
#     total, correct = [], []
#     for result in results.values():
#         result.sort()
#         passed = [r[1]["passed"] for r in result]
#         total.append(len(passed))
#         correct.append(sum(passed))
#     total = np.array(total)
#     correct = np.array(correct)

#     ks = k
#     pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
#                  for k in ks if (total >= k).all()}

#     # Finally, save the results in one file:
#     def combine_results():
#         for sample in stream_jsonl(sample_file):
#             task_id = sample["task_id"]
#             result = results[task_id].pop(0)
#             sample["result"] = result[1]["result"]
#             sample["passed"] = result[1]["passed"]
#             yield sample

#     out_file = sample_file + "_results.jsonl"
#     print(f"Writing results to {out_file}...")
#     write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

#     return pass_at_k

# ==========
# evaluate_functional_correctness.py




def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 1.0,
    problem_file: str = HUMAN_EVAL,
):

    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, [1], n_workers, timeout, problem_file)
    print(results)




def evaluate_functional_correctness(
    task_id,
    completion,
    problems,
    k: int = 1,
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
    executor: ThreadPoolExecutor = None,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    
    # import pdb;pdb.set_trace()

    # Check the generated samples against test suites.
    # with ThreadPoolExecutor(max_workers=n_workers) as executor:
    if True:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)


        args = (problems[task_id], completion, timeout, completion_id[task_id])
        future = executor.submit(check_correctness, *args)
        futures.append(future)
        completion_id[task_id] += 1
        n_samples += 1
        # import pdb;pdb.set_trace()

        # print(future)

        # assert len(completion_id) == len(problems) or len(completion_id)==1, "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))
        print(results)
    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    pass_at_k = estimate_pass_at_k(total, correct, k).mean()

    return pass_at_k, results

class CodeEvaluator:
    def judge(self, solution: str, ground_truth: dict) -> bool:
        raise NotImplementedError

class MBPPEVALEvaluator(CodeEvaluator):
    def judge(self, question: str, solution: str, ground_truth: dict, k: int) -> bool:
        # task_id = ground_truth["task_id"]
        # completion_code = self.extract_code(solution)

        # pass_at_k, judge_info = evaluate_functional_correctness(task_id, completion_code, self.problems, k)
        return 0.0, None
    

class HUMANEVALEvaluator(CodeEvaluator):
    def __init__(self,problem_file=HUMAN_EVAL, n_workers=1) -> None:
        super().__init__()
        self.problems = read_problems(problem_file)
        self.executor = ThreadPoolExecutor(max_workers=n_workers)

    def extract_code(self, text: str) -> str:
        # Use regex to find the content inside ```python\n...\n```
        matches = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        # Return the last match if any matches exist
        completion_code = matches[-1] if matches else ""
        

        # print("1==========")
        # print(completion_code)
        # print("2==========")


        # if "def" in completion_code:
        #     # Regex to match the function signature
        #     function_signature_pattern = r"def\s+.+\(.*?\).*:"
        #     # Regex to match the optional docstring
        #     docstring_pattern = r'^\s*("""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\')'

        #     # Find the function signature
        #     signature_match = re.search(function_signature_pattern, completion_code)
        #     if not signature_match:
        #         return completion_code

        #     # Find where the function signature ends
        #     signature_end = signature_match.end()
        #     remaining_code = completion_code[signature_end:]

        #     # Check for an optional docstring
        #     docstring_match = re.match(docstring_pattern, remaining_code, re.DOTALL)
        #     if docstring_match:
        #         # If a docstring is found, find where it ends
        #         docstring_end = docstring_match.end()
        #         completion_start = signature_end + docstring_end
        #     else:
        #         # No docstring, start directly after the function signature
        #         completion_start = signature_end

        #     # Extract the completion (everything after the signature and optional docstring)
        #     completion_code = completion_code[completion_start:].rstrip()
        # print(completion_code)
        # print("3==========")
        return completion_code

    
    def judge(self, question: str, solution: str, ground_truth: dict, k: int) -> bool:
        task_id = ground_truth["task_id"]
        completion_code = self.extract_code(solution)

        pass_at_k, judge_info = evaluate_functional_correctness(task_id, completion_code, self.problems, k, executor=self.executor)
        return pass_at_k, judge_info

evaluator_map = {}

def init_evaluator():
    global evaluator_map
    evaluator_map = {
        "humaneval": HUMANEVALEvaluator(),
        # "mbpp": MBPPEVALEvaluator
    }
if __name__ == "__main__":
   init_evaluator()
   question = " "
   solution = "```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n```"
   ground_truth = {
            "task_id": "HumanEval/0",
            "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
            "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
            "entry_point": "has_close_elements"
        }
   for i in range(100):
    pass_at_k, judge_info = evaluator_map["humaneval"].judge(question, solution, ground_truth, 1)
   print(pass_at_k)
   print(judge_info)