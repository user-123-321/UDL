import torch
from begin.algorithms.vcl.nodes import *
from begin.algorithms.bare.nodes import NCClassILBareTrainer, NCTaskILBareTrainer
from begin.algorithms.ewc.nodes import NCTaskILEWCTrainer, NCClassILEWCTrainer
from begin.algorithms.twp.nodes import NCTaskILTWPTrainer
from begin.algorithms.mas.nodes import NCTaskILMASTrainer
from begin.scenarios.nodes import NCScenarioLoader
from begin.utils.models_VCL import BGCNNode
from begin.utils import GCNNode
from begin.utils.loss_VCL import ELBO, BetaGenerator
from begin.utils.models_TWP import GCN as TWPGCN

dataset_name = "ogbn-arxiv"
num_tasks=8

with open("other_taskIL_results.txt", "a") as f:
    f.write(f"Running MAS on {dataset_name} with {num_tasks} tasks.\n")

full_results = {"acc": [], "AP": [], "AF": []}
for seed in range(5):
    print(f"Starting run {seed}/10")
    
    scenario = NCScenarioLoader(
        dataset_name=dataset_name,
        num_tasks=num_tasks,
        metric="accuracy",
        save_path="data",
        incr_type="task",
        task_shuffle=False,
    )
    
    model = GCNNode(
        in_feats=scenario.num_feats,
        n_classes=scenario.num_classes,
        n_hidden=256,
        dropout=0.0,
        n_layers=3,
    )
    
    benchmark = NCTaskILMASTrainer(
        model=model,
        scenario=scenario,
        optimizer_fn=lambda x: torch.optim.Adam(x, lr=1e-1, weight_decay=0),
        loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-1),
        device=torch.device("cuda:0"),
        scheduler_fn=lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(
            x,
            mode="min",
            patience=20,
            min_lr=1e-5,
            verbose=False,
        ),
        benchmark=True,
        seed=seed,
        batch_size=512,
    )
    
    results = benchmark.run(epoch_per_task=1000)
    
    acc = results["exp_test"]
    ap = results["exp_AP"]
    af = results["exp_AF"]
    
    full_results["acc"].append(acc)
    full_results["AP"].append(ap)
    full_results["AF"].append(af)
    
    with open("other_taskIL_results.txt", "a") as f:
        f.write(f"AP: {ap:.4f} | AF: {af:.4f} | accs: {acc}\n")
 

acc = torch.mean(torch.stack(full_results["acc"]), dim=0).tolist()
ap = torch.mean(torch.Tensor(full_results["AP"])).item()
af = torch.mean(torch.Tensor(full_results["AF"])).item()

with open("other_taskIL_results.txt", "a") as f:
    f.write(f"dataset {dataset_name} with {num_tasks} tasks full results:\n")
    f.write(f"AP: {ap:.4f} | AF: {af:.4f} | accs: {acc}\n\n\n\n")