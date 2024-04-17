import torch
import torch.nn.functional as F
from begin.trainers.nodes import NCTrainer, NCMinibatchTrainer
from begin.utils.models_VCL import BGCNNode
import copy
from begin.utils.utils_VCL import *
from collections import defaultdict
from begin.utils.loss_VCL import BetaGenerator
import dgl

class NCTaskILVCLTrainer(NCMinibatchTrainer):
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
        
        self.beta_generator.feed_full_dataset(torch.sum(curr_dataset.ndata["train_mask"]))
        self.beta_generator.feed_task(task_id)
        
        if task_id == 0:
            curr_training_states['coreset'] = []
            curr_training_states['coreset_size'] = 0
        else:
            if isinstance(curr_model, BGCNNode):
                curr_model.load_state_dict(curr_training_states['model_no_coreset'])
                curr_model.update_prior_and_reset()
        
        if self.coreset_size != 0:
            new_coreset, curr_dataset = self.coreset_method(curr_dataset, self.coreset_size, "task", "node")
            train_loader = dgl.dataloading.DataLoader(
                new_coreset,
                torch.nonzero(new_coreset.ndata['train_mask'], as_tuple=True)[0],
                dgl.dataloading.MultiLayerNeighborSampler([5, 10, 10]),
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
        input_nodes, output_nodes, blocks = _curr_batch
        blocks = [b.to(self.device) for b in blocks]
        labels = blocks[-1].dstdata['label']
        # print(labels.shape[0])
        outputs = model.bforward(blocks, blocks[0].srcdata['feat'], task_masks=blocks[-1].dstdata['task_specific_mask'])
        preds, kl = outputs[0], outputs[1]
        self.beta_generator.feed_batch(labels.shape[0])
        loss = self.loss_fn((preds, kl), labels, self.beta_generator.get())
        return {'preds': preds, 'loss': loss}

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

class NCClassILVCLTrainer(NCMinibatchTrainer):
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
        
        self.beta_generator.feed_full_dataset(torch.sum(curr_dataset.ndata["train_mask"]))
        self.beta_generator.feed_task(task_id)
        
        if task_id == 0:
            curr_training_states['coreset'] = []
            curr_training_states['coreset_size'] = 0
        else:
            if isinstance(curr_model, BGCNNode):
                curr_model.load_state_dict(curr_training_states['model_no_coreset'])
                curr_model.update_prior_and_reset()
        
        if self.coreset_size != 0:
            new_coreset, curr_dataset = self.coreset_method(curr_dataset, self.coreset_size, "task", "node")
            train_loader = dgl.dataloading.DataLoader(
                new_coreset,
                torch.nonzero(new_coreset.ndata['train_mask'], as_tuple=True)[0],
                dgl.dataloading.MultiLayerNeighborSampler([5, 10, 10]),
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
        input_nodes, output_nodes, blocks = _curr_batch
        blocks = [b.to(self.device) for b in blocks]
        labels = blocks[-1].dstdata['label']
        # print(labels.shape[0])
        outputs = model.bforward(blocks, blocks[0].srcdata['feat'])
        preds, kl = outputs[0], outputs[1]
        self.beta_generator.feed_batch(labels.shape[0])
        loss = self.loss_fn((preds, kl), labels, self.beta_generator.get())
        return {'preds': preds, 'loss': loss}
        
        # input_nodes, output_nodes, blocks = _curr_batch
        # blocks = [b.to(self.device) for b in blocks]
        # labels = blocks[-1].dstdata['label']
        # outputs = model.bforward(blocks, blocks[0].srcdata['feat'])
        # preds, kl = outputs[0], outputs[1]
        # self.beta_generator.feed_batch(labels.shape[0])
        # loss = self.loss_fn((preds, kl), labels, self.beta_generator.get())
        # return {'preds': preds, 'loss': loss}
        
        # curr_batch, mask = _curr_batch
        # outputs = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))
        # preds, kl = outputs[0][mask], outputs[1]
        # self.beta_generator.feed_batch(torch.sum(mask))
        # loss = self.loss_fn((preds, kl), curr_batch.ndata['label'][mask].to(self.device), self.beta_generator.get())
        # return {'preds': preds, 'loss': loss}

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

# class NCClassILEWCMinibatchTrainer(NCMinibatchTrainer):
#     def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
#         """
#             EWC needs `lamb`, the additional hyperparamter for the regularization term used in :func:`afterInference`
#         """
#         super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
#         self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 10000.
        
#     def afterInference(self, results, model, optimizer, _curr_batch, training_states):
#         """
#             The event function to execute some processes right after the inference step (for training).
#             We recommend performing backpropagation in this event function.
            
#             EWC performs regularization process in this function.
            
#             Args:
#                 results (dict): the returned dictionary from the event function `inference`.
#                 model (torch.nn.Module): the current trained model.
#                 optimizer (torch.optim.Optimizer): the current optimizer function.
#                 curr_batch (object): the data (or minibatch) for the current iteration.
#                 curr_training_states (dict): the dictionary containing the current training states.
                
#             Returns:
#                 A dictionary containing the information from the `results`.
#         """
#         loss_reg = 0
#         for _param, _fisher in zip(training_states['params'], training_states['fishers']):
#             for name, p in model.named_parameters():
#                 l = self.lamb * _fisher[name]
#                 l = l * ((p - _param[name]) ** 2)
#                 loss_reg = loss_reg + l.sum()
#         total_loss = results['loss'] + loss_reg
#         total_loss.backward()
#         optimizer.step()
#         return {'loss': total_loss.item(),
#                 'acc': self.eval_fn(self.predictionFormat(results), _curr_batch[-1][-1].dstdata['label'].to(self.device))}
    
#     def initTrainingStates(self, scenario, model, optimizer):
#         return {'fishers': [], 'params': []}
    
#     def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
#         """
#             The event function to execute some processes after training the current task.
            
#             EWC computes fisher information matrix and stores the learned weights to compute the penalty term in :func:`afterInference`
                
#             Args:
#                 task_id (int): the index of the current task.
#                 curr_dataset (object): The dataset for the current task.
#                 curr_model (torch.nn.Module): the current trained model.
#                 curr_optimizer (torch.optim.Optimizer): the current optimizer function.
#                 curr_training_states (dict): the dictionary containing the current training states.
#         """
#         super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
#         params = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
#         fishers = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
#         train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        
#         total_num_items = 0
#         for i, _curr_batch in enumerate(train_loader):
#             curr_model.zero_grad()
#             curr_results = self.inference(curr_model, _curr_batch, curr_training_states)
#             curr_results['loss'].backward()
#             curr_num_items =_curr_batch[-1][-1].dstdata['label'].shape[0]
#             total_num_items += curr_num_items
#             for name, p in curr_model.named_parameters():
#                 params[name] = p.data.clone().detach()
#                 fishers[name] += (p.grad.data.clone().detach() ** 2) * curr_num_items
                    
#         for name, p in curr_model.named_parameters():
#             fishers[name] /= total_num_items
                
#         curr_training_states['fishers'].append(fishers)
#         curr_training_states['params'].append(params)
        
# class NCDomainILEWCTrainer(NCClassILVCLTrainer):
#     """
#         This trainer has the same behavior as `NCClassILEWCTrainer`.
#     """
#     pass

# class NCTimeILEWCTrainer(NCClassILVCLTrainer):
#     """
#         This trainer has the same behavior as `NCClassILEWCTrainer`.
#     """
#     pass
