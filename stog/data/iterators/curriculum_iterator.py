from typing import Iterable
import logging
import random
import re
import numpy as np
from overrides import overrides

from stog.utils import lazy_groups_of
from stog.data.instance import Instance
from stog.data.iterators.bucket_iterator import BucketIterator, DataIterator
from stog.data.dataset import Batch
from stog.utils import logging

logger = logging.init_logger()


@DataIterator.register("CompetenceCurriculum")
class CompetenceCurriculumIterator(BucketIterator):
    def __init__(self,*args, curriculum_len=None, initial_competence=None, slope_power=2, damr_name='DAMRR0V2', **kwargs):
        assert int(curriculum_len) > 0
        assert float(initial_competence) > 0
        assert float(slope_power)
        super().__init__(*args,**kwargs)
        self.curriculum_len = curriculum_len
        self.initial_competence = initial_competence
        self.slope_power = slope_power
        self.initial_competence_powered = self.initial_competence ** self.slope_power
        self.timestep = None
        self.traindata = None
        self.traindata_difficulty = None
        self.damr_name = damr_name

    def _init_curriculum(self, instances):
        self.traindata = instances
        self.timestep = 0
        self.damr = globals()[self.damr_name](instances)
        self.traindata_difficulty = self.damr.compute_data_difficulty()


    def competence(self, timestep):
        return 1 if timestep >= self.curriculum_len else ( timestep * ( 1 - self.initial_competence_powered ) / self.curriculum_len + self.initial_competence_powered ) ** ( 1 / self.slope_power ) 

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool, train_epoch = None) -> Iterable[Batch]:
        if train_epoch is None:
            yield from super()._create_batches(instances, shuffle=shuffle)
            return
        
        if self.timestep is None:
            self._init_curriculum(instances)
        
        assert instances is self.traindata, 'curriculumiter cannot be used on other training data'

        if self.timestep >= self.curriculum_len: 
            yield from super()._create_batches(instances, shuffle=shuffle)
            return

        ninstances=0
        while True:
            current_competence = self.competence(self.timestep)
            sampled_data = [item for idx, item in enumerate(self.traindata) if self.traindata_difficulty[idx] <= current_competence + 0.0001]
            idx = list(range(len(sampled_data)))
            random.shuffle(idx)
            batch = [sampled_data[i] for i in idx[:self._batch_size]]
            yield Batch(batch)
            self.timestep += 1
            ninstances += self._batch_size
            if ninstances >= len(instances):
                break
        

class DAMR:
    def __init__(self,instances):
        self.data = instances
        self.amrs = [x.fields['amr'].metadata for x in instances]
        self.data_difficulty = None
        self.concept_idf=None
        self.rel_idf=None

    def compute_data_difficulty(self):
        if self.data_difficulty is None:
            raw = np.array([self.get_amr_difficulty(item) for item in self.amrs],dtype='float')
            cdf = (raw[None,:] <= raw[:,None]).mean(axis=1)
            self.data_difficulty = cdf
            self.raw_data_difficulty = raw
        return self.data_difficulty
        
    def get_amr_difficulty(self, amr): raise NotImplementedError

    def compute_concept_idf(self):
        df = {}
        for item in self.amrs:
            for nodeinfo in item.graph.get_nodes():
                concept = nodeinfo.instance
                if not concept: continue
                df[concept] = df.get(concept,0)+1
        self.concept_idf = {c:np.log(len(self.amrs)/v) for c,v in df.items()}

    def compute_rel_idf(self):
        df = {}
        for item in self.amrs:
            for edge,info in item.graph.get_edges.items():
                rel = info['label']
                df[rel] = df.get(rel, 0) + 1
        self.rel_idf = {c:np.log(len(self.amrs)/v) for c,v in df.items()}

    def rel_difficulty(self,rel):
        if self.rel_idf is None:
            self.compute_rel_idf()
        if 'ARG' in rel: return 1
        return 1 + self.rel_idf.get(rel, 1) 
    
    def concept_difficulty(self, concept):
        if self.concept_idf is None:
            self.compute_concept_idf()
        return 1 + self.concept_idf.get(concept, 1)


class DAMRV1(DAMR):
    def _get_nodes_depth(self, amr):
        depths = {amr.graph._top:1}
        while True:
            added=False
            for edge in amr.graph.get_edges():
                src = edge[0].identifier
                if src not in depths:continue
                depth = depths[src]
                tgt = edge[1].identifier
                if tgt not in depths:
                    depths[tgt] = depth + 1
                    added = True
            if not added: break
        # this should not happen   
        for nodeinfo in amr.graph.get_nodes():
            node = nodeinfo.instance
            if node not in depths:
                depths[node]=1
        return depths
    
    def get_amr_difficulty(self, amr):
        difficulty = 0
        nodes_depth = self._get_nodes_depth(amr)

        for nodeinfo in amr.graph.get_nodes():
            concept = nodeinfo.instance
            node = nodeinfo.identifier
            difficulty += self.concept_difficulty(concept) * nodes_depth[node] ** 2

        return difficulty


class DAMRV2(DAMR):

    def get_amr_difficulty(self, amr):
        difficulty = 0
        for edge,relinfo in amr.graph.get_edges().items():
            src_concept = edge[0].instance
            tgt_concept = edge[1].instance
            rel = relinfo['label']
            difficulty+= self.rel_difficulty(rel) * (
                self.concept_difficulty(src_concept) + self.concept_difficulty(tgt_concept))
        return difficulty

class DAMRR0V2(DAMRV2):
    def rel_difficulty(self,rel):
        if 'ARG' in rel: return 1
        return 2
    
    def concept_difficulty(self, concept):
        if self.concept_idf is None:
            self.compute_concept_idf()
        if re.match(r'^.*-\d+$',concept):
            return 1 + self.concept_idf.get(concept, 1)
        return 1

class NodeCountDAMR(DAMR):
    def get_amr_difficulty(self, amr): 
        return len(amr.graph.get_nodes())

class EdgeCountDAMR(DAMR):
    def get_amr_difficulty(self, amr): 
        return len(amr.graph.get_edges())


__all__ = [
    'CompetenceCurriculumIterator'
]