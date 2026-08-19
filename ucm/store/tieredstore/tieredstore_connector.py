
from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple



import torch



from ucm.logger import init_logger

from ucm.store.dramstore.dramstore_connector import UcmDramStore

from ucm.store.nfsstore.nfsstore_connector import UcmNfsStore

from ucm.store.ucmstore import Task, UcmKVStoreBase



logger = init_logger(__name__)



SUCCESS = 0

FAILURE = -1



# Scheduler/worker TieredStore instances live in the same vLLM EngineCore

# process in the current integration. Keep placement and DRAM reservations

# process-shared, matching UcmDramStore's shared-cache semantics.

_PROCESS_SHARED_PLACEMENT: Dict[str, str] = {}

_PROCESS_SHARED_DRAM_RESERVED: set[str] = set()





@dataclass

class TieredTask(Task):

    dram_task: Optional[Task] = None

    ssd_task: Optional[Task] = None

    transfer_ms: Optional[float] = None





class UcmTieredStore(UcmKVStoreBase):

    """Minimal DRAM-first, local-SSD-second KV store."""



    def __init__(self, config: Dict):

        super().__init__(config)



        dram_config = dict(config)

        dram_config["max_cache_size"] = int(

            config.get("dram_max_cache_size", 512 * 1024 * 1024)

        )



        ssd_config = dict(config)



        self.dram = UcmDramStore(dram_config)

        self.ssd = UcmNfsStore(ssd_config)



        self.placement = _PROCESS_SHARED_PLACEMENT

        self.dram_reserved = _PROCESS_SHARED_DRAM_RESERVED

        self.max_dram_blocks = self.dram.max_block_num



        logger.info(

            "UCM tiered store initialized: dram_max_blocks=%d "

            "dram_max_bytes=%d ssd_backend=%s",

            self.max_dram_blocks,

            self.dram.max_cache_byte,

            config.get("storage_backends"),

        )



    def cc_store(self) -> int:

        return 0



    def create(self, block_ids: List[str]) -> List[int]:

        results = [SUCCESS] * len(block_ids)

        new_dram: list[tuple[int, str]] = []

        new_ssd: list[tuple[int, str]] = []



        occupied = len(self.dram_reserved)



        for i, block_id in enumerate(block_ids):

            tier = self.placement.get(block_id)

            if tier is not None:

                continue



            # Recover placement if a store already contains the block.

            if self.dram.lookup([block_id])[0]:

                self.placement[block_id] = "dram"

                self.dram_reserved.add(block_id)

                occupied = len(self.dram_reserved)

                continue



            if self.ssd.lookup([block_id])[0]:

                self.placement[block_id] = "ssd"

                continue



            if occupied < self.max_dram_blocks:

                self.placement[block_id] = "dram"

                self.dram_reserved.add(block_id)

                occupied += 1

                new_dram.append((i, block_id))

            else:

                self.placement[block_id] = "ssd"

                new_ssd.append((i, block_id))



        if new_dram:

            ids = [x[1] for x in new_dram]

            rets = self.dram.create(ids)

            for (i, block_id), ret in zip(new_dram, rets):

                results[i] = ret

                if ret != SUCCESS:

                    self.placement.pop(block_id, None)

                    self.dram_reserved.discard(block_id)



        if new_ssd:

            ids = [x[1] for x in new_ssd]

            rets = self.ssd.create(ids)

            for (i, block_id), ret in zip(new_ssd, rets):

                results[i] = ret

                if ret != SUCCESS:

                    self.placement.pop(block_id, None)



        if new_dram or new_ssd:

            logger.info(

                "UCM_TIER_CREATE total=%d dram=%d ssd=%d "

                "dram_reserved=%d/%d",

                len(block_ids),

                len(new_dram),

                len(new_ssd),

                len(self.dram_reserved),

                self.max_dram_blocks,

            )



        return results



    def lookup(self, block_ids: List[str]) -> List[bool]:

        if not block_ids:

            return []



        dram_hits = self.dram.lookup(block_ids)

        result = list(dram_hits)



        miss_indices = [i for i, hit in enumerate(dram_hits) if not hit]

        ssd_hit_count = 0



        if miss_indices:

            miss_ids = [block_ids[i] for i in miss_indices]

            ssd_hits = self.ssd.lookup(miss_ids)



            for idx, block_id, hit in zip(

                miss_indices, miss_ids, ssd_hits

            ):

                if hit:

                    result[idx] = True

                    self.placement[block_id] = "ssd"

                    ssd_hit_count += 1



        for block_id, hit in zip(block_ids, dram_hits):

            if hit:

                self.placement[block_id] = "dram"

                self.dram_reserved.add(block_id)



        if ssd_hit_count:

            logger.info(

                "UCM_TIER_LOOKUP total=%d dram_hits=%d ssd_hits=%d",

                len(block_ids),

                sum(dram_hits),

                ssd_hit_count,

            )



        return result



    def prefetch(self, block_ids: List[str]) -> None:

        return None



    def _resolve_tiers(self, block_ids: List[str]) -> Dict[str, str]:

        unknown = list(

            dict.fromkeys(

                block_id

                for block_id in block_ids

                if block_id not in self.placement

            )

        )



        if unknown:

            dram_hits = self.dram.lookup(unknown)

            remaining = []



            for block_id, hit in zip(unknown, dram_hits):

                if hit:

                    self.placement[block_id] = "dram"

                    self.dram_reserved.add(block_id)

                else:

                    remaining.append(block_id)



            if remaining:

                ssd_hits = self.ssd.lookup(remaining)

                for block_id, hit in zip(remaining, ssd_hits):

                    if hit:

                        self.placement[block_id] = "ssd"

                    else:

                        raise KeyError(

                            f"TieredStore cannot resolve block {block_id}"

                        )



        return self.placement



    def load(

        self,

        block_ids: List[str],

        offset: List[int],

        dst_tensor: List[torch.Tensor],

    ) -> Task:

        self._resolve_tiers(block_ids)



        dram_ids, dram_offsets, dram_tensors = [], [], []

        ssd_ids, ssd_offsets, ssd_tensors = [], [], []



        logical_dram = set()

        logical_ssd = set()



        for block_id, off, tensor in zip(block_ids, offset, dst_tensor):

            if self.placement[block_id] == "dram":

                dram_ids.append(block_id)

                dram_offsets.append(off)

                dram_tensors.append(tensor)

                logical_dram.add(block_id)

            else:

                ssd_ids.append(block_id)

                ssd_offsets.append(off)

                ssd_tensors.append(tensor)

                logical_ssd.add(block_id)



        task = TieredTask()



        if dram_ids:

            task.dram_task = self.dram.load(

                dram_ids, dram_offsets, dram_tensors

            )

        if ssd_ids:

            task.ssd_task = self.ssd.load(

                ssd_ids, ssd_offsets, ssd_tensors

            )



        logger.info(

            "UCM_TIER_LOAD dram_blocks=%d ssd_blocks=%d",

            len(logical_dram),

            len(logical_ssd),

        )

        return task



    def dump(

        self,

        block_ids: List[str],

        offset: List[int],

        src_tensor: List[torch.Tensor],

    ) -> Task:

        missing = [

            block_id

            for block_id in set(block_ids)

            if block_id not in self.placement

        ]

        if missing:

            raise KeyError(

                f"TieredStore dump without placement: {missing[:3]}"

            )



        dram_ids, dram_offsets, dram_tensors = [], [], []

        ssd_ids, ssd_offsets, ssd_tensors = [], [], []



        logical_dram = set()

        logical_ssd = set()



        for block_id, off, tensor in zip(block_ids, offset, src_tensor):

            if self.placement[block_id] == "dram":

                dram_ids.append(block_id)

                dram_offsets.append(off)

                dram_tensors.append(tensor)

                logical_dram.add(block_id)

            else:

                ssd_ids.append(block_id)

                ssd_offsets.append(off)

                ssd_tensors.append(tensor)

                logical_ssd.add(block_id)



        task = TieredTask()



        if dram_ids:

            task.dram_task = self.dram.dump(

                dram_ids, dram_offsets, dram_tensors

            )

        if ssd_ids:

            task.ssd_task = self.ssd.dump(

                ssd_ids, ssd_offsets, ssd_tensors

            )



        logger.info(

            "UCM_TIER_DUMP dram_blocks=%d ssd_blocks=%d",

            len(logical_dram),

            len(logical_ssd),

        )

        return task



    def fetch_data(

        self,

        block_ids: List[str],

        offset: List[int],

        dst_addr: List[int],

        size: List[int],

    ) -> Task:

        raise NotImplementedError



    def dump_data(

        self,

        block_ids: List[str],

        offset: List[int],

        src_addr: List[int],

        size: List[int],

    ) -> Task:

        raise NotImplementedError



    def wait(self, task: TieredTask) -> int:

        ret = SUCCESS

        transfer_ms = 0.0

        have_timing = False



        if task.dram_task is not None:

            dram_ret = self.dram.wait(task.dram_task)

            if dram_ret != SUCCESS:

                ret = FAILURE



            dram_ms = getattr(task.dram_task, "transfer_ms", None)

            if dram_ms is not None:

                transfer_ms += float(dram_ms)

                have_timing = True



        if task.ssd_task is not None:

            ssd_ret = self.ssd.wait(task.ssd_task)

            if ssd_ret != SUCCESS:

                ret = FAILURE



        task.transfer_ms = transfer_ms if have_timing else None

        return ret



    def commit(

        self, block_ids: List[str], is_success: bool = True

    ) -> None:

        dram_ids = [

            block_id

            for block_id in block_ids

            if self.placement.get(block_id) == "dram"

        ]

        ssd_ids = [

            block_id

            for block_id in block_ids

            if self.placement.get(block_id) == "ssd"

        ]



        if dram_ids:

            self.dram.commit(dram_ids, is_success)

        if ssd_ids:

            self.ssd.commit(ssd_ids, is_success)



        if not is_success:

            for block_id in block_ids:

                tier = self.placement.pop(block_id, None)

                if tier == "dram":

                    self.dram_reserved.discard(block_id)



    def check(self, task: Task) -> Tuple[int, bool]:

        return SUCCESS, True

