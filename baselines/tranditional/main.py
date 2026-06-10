import argparse
import csv
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from models.kge_models import TransE, DistMult
from data.dataloader import KGData, TrainDataset, EvalDataset, get_dataloader

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
        
        # Load Data WITH Inverse Relations (Matching HoGRN protocol)
        # This doubles the relations and allows Head prediction via Tail prediction on inverse triples
        print(f"Loading data from {args.data_path}...")
        self.kg_data = KGData(args.data_path, add_inverse=True) 
        self.args.num_ent = self.kg_data.num_ent
        self.args.num_rel = self.kg_data.num_rel
        
        print(f"Dataset Loaded. Num Ent: {self.args.num_ent}, Num Rel: {self.args.num_rel}")
        
        # Build Model
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        
        # Use BCELoss (Matching HoGRN protocol)
        # We will use pointwise independent prediction (1-vs-All)
        self.criterion = nn.BCELoss()
        
        # Checkpoints dir
        self.ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # Train DataLoader
        # Use TrainDataset (1-vs-All)
        # This dataset returns ((h, r), multihot_label)
        train_dataset = TrainDataset(self.kg_data.train_sr2o, self.args.num_ent)
        self.train_loader = get_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
        
    def _build_model(self):
        if self.args.model == 'TransE':
            return TransE(self.args)
        elif self.args.model == 'DistMult':
            return DistMult(self.args)
        elif self.args.model == 'RotatE':
            # RotatE in kge_models.py but not imported or defined in models list previously?
            # It was in the file, we need to import it.
            # Let's assume it's imported (I will update imports below if needed)
            from models.kge_models import RotatE 
            return RotatE(self.args)
        elif self.args.model == 'ComplEx':
            from models.kge_models import ComplEx
            return ComplEx(self.args)
        elif self.args.model == 'ConvE':
            from models.kge_models import ConvE
            return ConvE(self.args)
        elif self.args.model == 'TuckER':
            from models.kge_models import TuckER
            return TuckER(self.args)
        else:
            raise ValueError(f"Unknown model: {self.args.model}")

    def train(self):
        best_mrr = 0.0
        kill_cnt = 0
        
        for epoch in range(1, self.args.max_epochs + 1):
            # print("########")
            t0 = time.time()
            
            self.model.train()
            total_loss = 0
            
            # Using 1-vs-All DataLoader
            # batch: (inputs, labels)
            # inputs: (Batch, 2) -> [h, r]
            # labels: (Batch, NumEnt) -> Multi-hot
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                h, r = inputs[:, 0], inputs[:, 1]
                
                # Forward (Returns Sigmoid Scores)
                # (Batch, NumEnt)
                preds = self.model(h, r)
                
                loss = self.criterion(preds, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            # print("Time cost in one epoch for training: {:.4f}s".format((time.time()-t0)))
            
            # Evaluation
            if epoch % self.args.eval_freq == 0:
                val_metrics = self.evaluate(split='valid', epoch=epoch)
                current_mrr = val_metrics['mrr']
                
                if current_mrr > best_mrr:
                    best_mrr = current_mrr
                    kill_cnt = 0
                    save_path = os.path.join(self.ckpt_dir, f"best_model_{self.args.model}_{self.args.dataset}.pth")
                    torch.save(self.model.state_dict(), save_path)
                else:
                    kill_cnt += 1
                    if kill_cnt >= self.args.patience:
                        print("Early Stopping!!")
                        print('[Epoch {}]: Training Loss: {:.5}, Best Valid MRR: {:.5}\n\n'.format(epoch, avg_loss, best_mrr))
                        break
                
                print('[Epoch {}]: Training Loss: {:.5}, Best Valid MRR: {:.5}\n\n'.format(epoch, avg_loss, best_mrr))
            else:
                print('[Epoch {}]: Training Loss: {:.5}\n'.format(epoch, avg_loss))

    def _run_eval_pass(self, eval_triples):
        """Runs filtered, tie-aware, full-entity ranking over a list of (h, r, t) queries."""
        hits1, hits3, hits10, mrr = 0, 0, 0, 0
        total = 0

        eval_dataset = EvalDataset(eval_triples, self.kg_data.all_sr2o, self.args.num_ent)
        data_loader = DataLoader(eval_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        with torch.no_grad():
            for batch in data_loader:
                triples = batch[0].to(self.device)
                labels = batch[1].to(self.device) # Multi-hot of ALL known tails

                h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]

                # Get scores for all entities
                scores = self.model(h, r)

                # Filtered Setting
                b_range = torch.arange(scores.size(0), device=self.device)
                target_score = scores[b_range, t]

                # Mask all true tails to -1e7
                scores = scores.masked_fill(labels.bool(), -1e7)
                scores[b_range, t] = target_score

                # Tie-aware rank (average rank among ties):
                # rank = greater + (equal + 1) / 2
                # where equal includes the target entity itself.
                greater = (scores > target_score.unsqueeze(1)).sum(dim=1).float()
                equal = (scores == target_score.unsqueeze(1)).sum(dim=1).float()
                rank = greater + (equal + 1.0) / 2.0

                mrr += (1.0 / rank).sum().item()
                hits1 += (rank <= 1).sum().item()
                hits3 += (rank <= 3).sum().item()
                hits10 += (rank <= 10).sum().item()
                total += scores.size(0)

        return {
            'mrr': mrr / total,
            'h1': hits1 / total,
            'h3': hits3 / total,
            'h10': hits10 / total,
        }

    def evaluate(self, split='valid', epoch=0, label=None):
        self.model.eval()

        queries = getattr(self.kg_data, f"{split}_triples")

        # Unified evaluation protocol:
        # 1) Bidirectional evaluation via inverse relation queries (tail pred + head pred via inverse)
        # 2) Filtered setting
        # 3) Tie-aware ranking
        # 4) Full-entity ranking (score against all entities)

        # Calculate offset for inverse relations: num_base_rel = num_rel / 2
        num_base_rel = self.args.num_rel // 2

        tail_queries = list(queries)
        head_queries = [(t, r + num_base_rel, h) for h, r, t in queries] if self.kg_data.add_inverse else []

        tail_results = self._run_eval_pass(tail_queries)
        head_results = self._run_eval_pass(head_queries) if head_queries else tail_results

        results = {}
        for key in ('mrr', 'h1', 'h3', 'h10'):
            results[f'{key}_tail'] = tail_results[key]
            results[f'{key}_head'] = head_results[key]
            results[f'{key}_avg'] = (tail_results[key] + head_results[key]) / 2.0
        # Backwards-compatible aliases pointing at the bidirectional average
        results['mrr'] = results['mrr_avg']
        results['h1'] = results['h1_avg']
        results['h3'] = results['h3_avg']
        results['h10'] = results['h10_avg']

        eval_label = label or split
        print('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(
            epoch, eval_label, results['mrr_tail'], results['mrr_head'], results['mrr_avg']))
        for k, key in ((1, 'h1'), (3, 'h3'), (10, 'h10')):
            print('[Epoch {} {}]: Hits@{}: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(
                epoch, eval_label, k, results[f'{key}_tail'], results[f'{key}_head'], results[f'{key}_avg']))

        return results

if __name__ == "__main__":
    METRICS_CSV_HEADER = [
        "Dataset", "Model",
        "MRR_Tail", "MRR_Head", "MRR_Avg",
        "Hits@1_Tail", "Hits@1_Head", "Hits@1_Avg",
        "Hits@3_Tail", "Hits@3_Head", "Hits@3_Avg",
        "Hits@10_Tail", "Hits@10_Head", "Hits@10_Avg",
        "seconds",
    ]

    def append_metrics_csv(output_path, dataset, model, metrics, seconds):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_header = not os.path.exists(output_path)
        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(METRICS_CSV_HEADER)
            writer.writerow([
                dataset,
                model,
                f"{metrics['mrr_tail']:.5f}", f"{metrics['mrr_head']:.5f}", f"{metrics['mrr_avg']:.5f}",
                f"{metrics['h1_tail']:.5f}", f"{metrics['h1_head']:.5f}", f"{metrics['h1_avg']:.5f}",
                f"{metrics['h3_tail']:.5f}", f"{metrics['h3_head']:.5f}", f"{metrics['h3_avg']:.5f}",
                f"{metrics['h10_tail']:.5f}", f"{metrics['h10_head']:.5f}", f"{metrics['h10_avg']:.5f}",
                f"{seconds:.3f}",
            ])
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../datasets/FB15K-237")
    parser.add_argument("--dataset", type=str, default="FB15K-237")
    parser.add_argument("--model", type=str, default="TransE")
    parser.add_argument("--gpu", type=int, default=0)
    
    # Defaults matching HoGRN 1-vs-All settings
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--emb_dim", type=int, default=100)
    parser.add_argument("--margin", type=float, default=40.0, help="Gamma for TransE")
    parser.add_argument("--l2", type=float, default=0.0, help="Weight Decay")
    
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--patience", type=int, default=25)
    
    args = parser.parse_args()

    run_start = time.perf_counter()
    trainer = Trainer(args)
    trainer.train()

    print("\nLoading best model for final evaluation...")
    ckpt_path = os.path.join(trainer.ckpt_dir, f"best_model_{args.model}_{args.dataset}.pth")
    if os.path.exists(ckpt_path):
        trainer.model.load_state_dict(torch.load(ckpt_path, map_location=trainer.device))
        print("Evaluating on test set...")
        final_metrics = trainer.evaluate(split='test', epoch=0, label='test')
        run_seconds = time.perf_counter() - run_start
        print(
            "FINAL_EVAL_METRICS baseline=traditional model={} dataset={} split=test "
            "mrr_tail={:.5f} mrr_head={:.5f} mrr_avg={:.5f} "
            "h1_tail={:.5f} h1_head={:.5f} h1_avg={:.5f} "
            "h3_tail={:.5f} h3_head={:.5f} h3_avg={:.5f} "
            "h10_tail={:.5f} h10_head={:.5f} h10_avg={:.5f}".format(
                args.model, args.dataset,
                final_metrics['mrr_tail'], final_metrics['mrr_head'], final_metrics['mrr_avg'],
                final_metrics['h1_tail'], final_metrics['h1_head'], final_metrics['h1_avg'],
                final_metrics['h3_tail'], final_metrics['h3_head'], final_metrics['h3_avg'],
                final_metrics['h10_tail'], final_metrics['h10_head'], final_metrics['h10_avg'],
            )
        )
        print("RUNTIME_STD baseline=traditional model={} dataset={} seconds={:.3f}".format(
            args.model, args.dataset, run_seconds))
        metrics_root = os.environ.get("SPARSEKGC_OUTPUT_DIR")
        metrics_path = (
            os.path.join(metrics_root, "traditional_metrics.csv")
            if metrics_root
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), "timings", "traditional_metrics.csv")
        )
        append_metrics_csv(
            metrics_path,
            args.dataset,
            args.model,
            final_metrics,
            run_seconds,
        )
    else:
        print("No best model checkpoint found.")
