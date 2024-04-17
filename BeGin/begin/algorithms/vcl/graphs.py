import torch
import torch.nn.functional as F
from begin.trainers.graphs import GCTrainer
from begin.utils.models_VCL import BCGNGraph
import copy
from begin.utils.utils_VCL import *
from collections import defaultdict
from begin.utils.loss_VCL import BetaGenerator
import dgl

class GCTaskILVCLTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, coreset_size=0, coreset_method=coreset_rand, num_train_val_samples=10, num_test_samples=100, beta_generator=BetaGenerator(), **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
    
        self.coreset_size = coreset_size
        self.coreset_method = coreset_method
        self.num_train_val_samples = num_train_val_samples
        self.num_test_samples = num_test_samples
        self.beta_generator = beta_generator
     
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        
        curr_model.update_num_samples(self.num_train_val_samples)
        self.beta_generator.feed_full_dataset(len(curr_dataset['train']))
        self.beta_generator.feed_task(task_id)
        
        if task_id == 0:
            curr_training_states['coreset'] = []
            curr_training_states['coreset_size'] = 0
        else:
            if isinstance(curr_model, BCGNGraph):
                curr_model.load_state_dict(curr_training_states['model_no_coreset'])
                curr_model.update_prior_and_reset()
        
        if self.coreset_size != 0:
            new_coreset, curr_dataset = self.coreset_method(curr_dataset, self.coreset_size, "task", "graph")
            train_loader = dgl.dataloading.GraphDataLoader(
                new_coreset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                worker_init_fn=self._dataloader_seed_worker,
                generator=torch.Generator(),
            )
            curr_training_states['coreset'].append(train_loader)
            curr_training_states['coreset_size'] += self.coreset_size
            
        print(f"Task {task_id} non-coreset training started.")
        
    def inference(self, model, _curr_batch, training_states):
        graphs, labels, masks = _curr_batch
        outputs = model(graphs.to(self.device),
                        graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                        edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                        edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None,
                        task_masks=masks)
        loss = self.loss_fn(outputs, labels.to(self.device))
        return {'preds': outputs[0], 'loss': loss}

    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        curr_training_states['model_no_coreset'] = copy.deepcopy(curr_model.state_dict())
        
        was_training = curr_model.training

        # train on coreset
        if self.coreset_size != 0:
            curr_model.load_state_dict(curr_training_states['best_weights'])
            curr_model.update_prior_and_reset()
                
            curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
            curr_training_states['best_val_loss'] = 1e10
            self._reset_optimizer(curr_optimizer)
            
            self.beta_generator.feed_full_dataset(curr_training_states['coreset_size'])
                
            stop_training = False
            print(f"Task {task_id} coreset training started.") # NOTE doesn't show validation accuracy
            for epoch in range(self.max_num_epochs):
                if stop_training: print("Stopped early"); continue
                train_stats = {}
                
                curr_model.train()
                for idx in torch.randperm(len(curr_training_states['coreset'])):
                    for batch in curr_training_states['coreset'][idx]:
                        self._trainWrapper(curr_model, curr_optimizer, batch, curr_training_states, train_stats)
                reduced_train_stats = self._reduceTrainingStats(train_stats)
                
                if self.verbose:
                    self.processTrainingLogs(task_id, epoch, 0., reduced_train_stats, defaultdict(lambda: 0.))
                
                curr_iter_results = {'val_metric': 0., 'train_stats': reduced_train_stats, 'val_stats': defaultdict(lambda: 0.)}
                if not self.processAfterEachIteration(curr_model, curr_optimizer, curr_training_states, curr_iter_results):
                    stop_training = True
                
        curr_model.update_num_samples(self.num_test_samples)
        if was_training:
            curr_model.train()
        else:
            curr_model.eval()
        
    def processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        return super().processAfterEachIteration(curr_model, curr_optimizer, curr_training_states, curr_iter_results)
        # return True