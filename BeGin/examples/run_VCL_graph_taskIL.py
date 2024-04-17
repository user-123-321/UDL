import torch
from begin.algorithms.vcl.graphs import *
from begin.scenarios.graphs import GCScenarioLoader
from begin.utils.models_VCL import BCGNGraph
from begin.utils.loss_VCL import ELBO, BetaGenerator

dataset_name = "mnist"
num_tasks = 5
coreset_size = 1000

with open("VCL_graph_taskIL_results.txt", "a") as f:
    f.write(f"Starting VCL with coreset {coreset_size} on {dataset_name} GC Task IL with {num_tasks} tasks\n")

full_results = {"acc": [], "AP": [], "AF": []}
for seed in range(10):
    print(f"Starting run {seed}/10")
    
    scenario = GCScenarioLoader(
        dataset_name=dataset_name,
        num_tasks=num_tasks,
        metric="accuracy",
        save_path="data",
        incr_type="task",
        task_shuffle=False,
    )
    
    model = BCGNGraph(
        in_feats=scenario.num_feats,
        n_classes=scenario.num_classes,
        n_hidden=146,
        dropout=0.0,
        n_layers=4,
    )
    
    benchmark = GCTaskILVCLTrainer(
        model=model,
        scenario=scenario,
        optimizer_fn=lambda x: torch.optim.Adam(x, lr=1e-2, weight_decay=0),
        loss_fn=ELBO(),
        beta_generator=BetaGenerator(
            full_dataset=True,
            task_adaptive=False, 
            zero_on_first_task=False,
        ),
        coreset_size=coreset_size,
        num_train_val_samples=10,
        num_test_samples=100,
        device=torch.device("cuda:0"),
        scheduler_fn=lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(
            x,
            mode="min",
            patience=5,
            min_lr=1e-5,
            verbose=False,
        ),
        benchmark=True,
        seed=seed,
        batch_size=128,
    )
    
    results = benchmark.run(epoch_per_task=100)
    
    acc = results["exp_test"]
    ap = results["exp_AP"]
    af = results["exp_AF"]
    
    full_results["acc"].append(acc)
    full_results["AP"].append(ap)
    full_results["AF"].append(af)
    
    with open("VCL_graph_taskIL_results.txt", "a") as f:
        f.write(f"dataset {dataset_name} with {num_tasks} tasks\nAP: {ap:.4f} | AF: {af:.4f} | accs: {acc}\n")
 

acc = torch.mean(torch.stack(full_results["acc"]), dim=0).tolist()
ap = torch.mean(torch.Tensor(full_results["AP"])).item()
af = torch.mean(torch.Tensor(full_results["AF"])).item()

with open("VCL_graph_taskIL_results.txt", "a") as f:
    f.write(f"dataset {dataset_name} with {num_tasks} tasks full results\nAP: {ap:.4f} | AF: {af:.4f} | accs: {acc}\n\n\n\n")
        