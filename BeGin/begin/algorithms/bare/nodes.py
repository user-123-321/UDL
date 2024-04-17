import sys
from begin.trainers.nodes import NCTrainer, NCMinibatchTrainer

class NCTaskILBareTrainer(NCMinibatchTrainer):
    def inference(self, model, _curr_batch, training_states):
        """
            The event function to execute inference step.
            For task-IL, we need to additionally consider task information for the inference step.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the inference results, such as prediction result and loss.
        """
        input_nodes, output_nodes, blocks = _curr_batch
        blocks = [b.to(self.device) for b in blocks]
        labels = blocks[-1].dstdata['label']
        preds = model.bforward(blocks, blocks[0].srcdata['feat'], task_masks=blocks[-1].dstdata['task_specific_mask'])
        loss = self.loss_fn(preds, labels)        
        return {'preds': preds, 'loss': loss}
    
class NCClassILBareTrainer(NCTrainer):
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    pass

class NCClassILBareMinibatchTrainer(NCMinibatchTrainer):
    """
        This trainer has the same behavior as `NCMinibatchTrainer`.
    """
    pass

class NCDomainILBareTrainer(NCTrainer):
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    pass
        
class NCTimeILBareTrainer(NCTrainer):
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    pass