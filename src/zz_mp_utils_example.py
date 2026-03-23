from time import sleep

from multiprocess.queues import Queue
from tqdm import tqdm

from utils.mp_utils import PoolWithTqdm, PoolWithTqdmSingle


def single_process(
    args_with_queue: tuple[int, Queue],
):
    with PoolWithTqdmSingle(args_with_queue) as (args, pos):
        work_item = args[0]
        for _ in tqdm(
            range(10 + pos * 2),
            desc=f"Processing {work_item}",
            position=pos,
            leave=False,
            mininterval=1,
        ):
            sleep(0.2)

    return work_item


def test_multiprocess_with_tqdm():
    num_processes = 4
    args = [(i,) for i in range(10)]
    with PoolWithTqdm(num_processes=num_processes, args=args) as tqdm_pool:
        results = []
        for result in tqdm(
            tqdm_pool.pool.imap_unordered(single_process, tqdm_pool.args_with_queue),
            total=len(tqdm_pool.args_with_queue),
            desc="Processing items",
            position=0,
            leave=False,
        ):
            results.append(result)

    print("All processed results:", results)


if __name__ == "__main__":
    test_multiprocess_with_tqdm()
