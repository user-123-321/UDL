import torch
from begin.algorithms.vcl.nodes import *
from begin.algorithms.bare.nodes import NCClassILBareTrainer, NCTaskILBareTrainer
from begin.algorithms.ewc.nodes import NCTaskILEWCTrainer, NCClassILEWCTrainer
from begin.scenarios.nodes import NCScenarioLoader
from begin.utils.models_VCL import BGCNNode
from begin.utils import GCNNode
from begin.utils.loss_VCL import ELBO, BetaGenerator

# results is a dict with 'exp_test' accuracy matrix, 'exp_AP', and 'exp_AF'

# for lr in [1e-3, 5e-3, 1e-2]:
# for coreset_size in [0, 5, 10, 15]:
# for full_dataset in [True, False]: # only matters for coreset training
# for task_adaptive in [True, False]:
#     for zero_on_first_task in [True, False]:
#         if task_adaptive or not zero_on_first_task:

dataset_name = "ogbn-arxiv"
num_tasks=8

full_results = {"acc": [], "AP": [], "AF": []}
for seed in range(10):
    print(f"Starting run {seed}/10")
    
    scenario = NCScenarioLoader(
        dataset_name=dataset_name,
        num_tasks=num_tasks,
        metric="accuracy",
        save_path="data",
        incr_type="task",
        task_shuffle=False,
    )
    
    model = BGCNNode(
        in_feats=scenario.num_feats,
        n_classes=scenario.num_classes,
        n_hidden=256,
        dropout=0.0,
        n_layers=3,
    )
    
    benchmark = NCTaskILVCLTrainer(
        model=model,
        scenario=scenario,
        optimizer_fn=lambda x: torch.optim.Adam(x, lr=1e-1, weight_decay=0),
        loss_fn=ELBO(),
        beta_generator=BetaGenerator(
            full_dataset=True,
            task_adaptive=False, 
            zero_on_first_task=False,
        ),
        coreset_size=0,
        num_train_val_samples=10,
        num_test_samples=100,
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
    
    with open("VCL_taskIL_results.txt", "a") as f:
        f.write(f"dataset {dataset_name} with {num_tasks} tasks | AP: {ap:.4f} | AF: {af:.4f} | accs: {acc}\n")
 

acc = torch.mean(torch.stack(full_results["acc"]), dim=0).tolist()
ap = torch.mean(torch.Tensor(full_results["AP"])).item()
af = torch.mean(torch.Tensor(full_results["AF"])).item()

with open("VCL_taskIL_results.txt", "a") as f:
    f.write(f"dataset {dataset_name} with {num_tasks} tasks full results | AP: {ap:.4f} | AF: {af:.4f} | accs: {acc}\n\n\n\n")
        
# from skopt import gp_minimize
# from skopt.space import Real, Integer
# from skopt.utils import use_named_args

# space = [
#     Real(1e-6, 1e-1, name='lr', prior='log-uniform'),
#     Real(0, 1e-3, name='weight_decay'),
#     Integer(0, 20, name='coreset_size'),
# ]

# full_results = {}

# @use_named_args(space)
# def objective(**params):
#     obj_results = {"acc": [], "AP": [], "AF": []}
    
#     for seed in range(1):
#         scenario = NCScenarioLoader(
#             dataset_name="citeseer",
#             num_tasks=3,
#             metric="accuracy",
#             save_path="data",
#             incr_type="class",
#             task_shuffle=False,
#         )
        
#         model = BGCNNode(
#             in_feats=scenario.num_feats,
#             n_classes=scenario.num_classes,
#             n_hidden=256,
#             dropout=0.0,
#             n_layers=3,
#         )
        
#         benchmark = NCClassILVCLTrainer(
#             model=model,
#             scenario=scenario,
#             optimizer_fn=lambda x: torch.optim.Adam(x, lr=params['lr'], weight_decay=params['weight_decay']),
#             loss_fn=ELBO(),
#             beta_generator=BetaGenerator(),
#             coreset_size=params['coreset_size'],
#             num_train_val_samples=10,
#             num_test_samples=100,
#             device=torch.device("cuda:0"),
#             scheduler_fn=lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 x,
#                 mode="min",
#                 patience=50,
#                 min_lr=1e-3 * 0.001 * 2.,
#                 verbose=False,
#             ),
#             benchmark=True,
#             seed=seed,
#         )
        
#         results = benchmark.run(epoch_per_task=10)
#         obj_results["acc"].append(results["exp_test"])
#         obj_results["AP"].append(results["exp_AP"])
#         obj_results["AF"].append(results["exp_AF"])
    
#     obj_str = f"lr: {params['lr']:.06f} | weight decay: {params['weight_decay']:.03f} | coreset size: {params['coreset_size']}"
    
#     full_results[obj_str] = {
#         "acc": torch.mean(torch.stack(obj_results["acc"]), dim=0).tolist(),
#         "AP": torch.mean(torch.Tensor(obj_results["AP"])).item(),
#         "AF": torch.mean(torch.Tensor(obj_results["AF"])).item()
#     }
    
#     with open("VCL_citeseer_classIL_results.txt", "a") as f:
#         f.write(f'{obj_str} | AP: {full_results[obj_str]["AP"]} | AF: {full_results[obj_str]["AF"]} | accs: {full_results[obj_str]["acc"]}')
    
#     return -full_results[obj_str]["AP"]

# res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)

# print("full results:", full_results)

# print(f"Best hyperparameters: {res_gp.x}")
