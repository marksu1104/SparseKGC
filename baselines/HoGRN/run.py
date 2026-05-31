from helper import *
from data_loader import *
from model.models import *

import os, time, argparse, csv, pickle
<<<<<<< HEAD
from pathlib import Path
=======
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
import numpy as np
import torch

class Runner(object):

    def load_data(self):
        """
        Read in raw triplets and convert them into a standard format. 
        """

        # Build the mapping table from all the data
<<<<<<< HEAD
        def parse_triple(line):
            parts = line.strip().split('\t')[:3]
            if len(parts) < 3:
                raise ValueError(f"Malformed triple line: {line!r}")
            # Most datasets in this workspace are h, tail, relation. The legacy
            # HoGRN WN18RR copy is h, relation, tail, so normalize it here.
            if self.p.dataset == 'WN18RR' and parts[1].startswith('_'):
                sub, rel, obj = parts
                return sub.lower(), obj.lower(), rel.lower()
            sub, obj, rel = parts
            return sub.lower(), obj.lower(), rel.lower()

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, obj, rel = parse_triple(line)
=======
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, obj, rel = map(str.lower, line.strip().split('\t')[:3])
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent		= len(self.ent2id)
        self.p.num_rel		= len(self.rel2id) // 2
        print("Dataset: ", self.p.dataset)
        print("NUM_ENT: ", self.p.num_ent)
        print("NUM_REL: ", self.p.num_rel)
        self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        # Use UIDs to represent entities and relationships in the data, and inverse relationships are used to expand the training set
        self.data = ddict(list)
        sr2o = ddict(set) 
        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
<<<<<<< HEAD
                sub, obj, rel = parse_triple(line)
=======
                sub, obj, rel = map(str.lower, line.strip().split('\t')[:3])
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train': 
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.data = dict(self.data) 

        self.sr2o = {k: list(v) for k, v in sr2o.items()} # train
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()} # train+valid+test

        self.triples  = ddict(list)
        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]: 
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        for sub, rel, obj in self.data['train']:
            rel_inv = rel + self.p.num_rel
            self.triples['train_tail'].append({'triple': (sub, rel, obj),      'label': self.sr2o_all[(sub, rel)]})
            self.triples['train_head'].append({'triple': (obj, rel_inv, sub),  'label': self.sr2o_all[(obj, rel_inv)]})
        
        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return  DataLoader(
                    dataset_class(self.triples[split], self.p),
                    batch_size      = batch_size,
                    shuffle         = shuffle,
                    num_workers     = max(0, self.p.num_workers),
                    collate_fn      = dataset_class.collate_fn
                )

        self.data_iter = {
            'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size, shuffle=True),
            'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size, shuffle=False),
            'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size, shuffle=False),
            'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size, shuffle=False),
            'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size, shuffle=False),
            'train_head':   get_data_loader(TestDataset,  'train_head',   self.p.batch_size, shuffle=False),
            'train_tail':   get_data_loader(TestDataset,  'train_tail',   self.p.batch_size, shuffle=False),
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """
        Construct the adjacency matrix for GCN.
        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index	= torch.LongTensor(edge_index).to(self.device).t()
        edge_type	= torch.LongTensor(edge_type). to(self.device)

        return edge_index, edge_type
    
    def __init__(self, params):
        """
        Constructor of the runner class.
        """
        self.p			= params
        self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model        = self.add_model(self.p.model, self.p.score_func)
        self.optimizer    = self.add_optimizer(self.model.parameters())

    def add_model(self, model, score_func):
        """
        Create the computational graph.
        """
        model_name = '{}_{}'.format(model, score_func)

        if   model_name.lower()	== 'hogrn_transe': 		model = HoGRN_TransE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower()	== 'hogrn_distmult': 	model = HoGRN_DistMult(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower()	== 'hogrn_conve': 		model = HoGRN_ConvE(self.edge_index, self.edge_type, params=self.p)
        else: raise NotImplementedError

        model.to(self.device)
        print("Model have {:.4f}M paramerters in total".format(sum(x.numel()/1e6 for x in model.parameters())))
        return model

    def add_optimizer(self, parameters):
        """
        Create an optimizer for training the parameters
        """
        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU
        """
        if split == 'train':
            triple, label = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
        -------
        """
        state = {
            'state_dict'	: self.model.state_dict(),
            'best_val'		: self.best_val,
            'best_epoch'	: self.best_epoch,
            'optimizer'		: self.optimizer.state_dict(),
            'args'			: vars(self.p)
        }
<<<<<<< HEAD
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
=======
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model
        """
        state				= torch.load(load_path)
        state_dict			= state['state_dict']
        self.best_val		= state['best_val']
        self.best_val_mrr	= self.best_val['mrr'] 

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])
    
    def load_checkpoint_args(self, load_path):
        """
        Load arguments from a saved checkpoint
        """
        state = torch.load(load_path)
        return state.get('args', None)

<<<<<<< HEAD
    def evaluate(self, split, epoch, emit_detail=True):
=======
    def evaluate(self, split, epoch):
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count
        
        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        left_results  = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results       = get_combined_results(left_results, right_results)
<<<<<<< HEAD
        log_split = 'holdout' if split == 'test' else split
        if emit_detail:
            self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, log_split, results['left_mrr'], results['right_mrr'], results['mrr']))
            self.logger.info('[Epoch {} {}]: MR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, log_split, results['left_mr'], results['right_mr'], results['mr']))
        # for k in range(10):
        # 	self.logger.info('[Epoch {} {}]: Hit@{}: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, log_split, k+1, results['left_hits@{}'.format(k+1)], results['right_hits@{}'.format(k+1)], results['hits@{}'.format(k+1)]))
        if emit_detail and split == 'test':
            for k in range(10):
                self.logger.info('[Epoch {} {}]: Hit@{}: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, log_split, k+1, results['left_hits@{}'.format(k+1)], results['right_hits@{}'.format(k+1)], results['hits@{}'.format(k+1)]))

        if emit_detail:
            self.logger.info(
                'EVAL_STD model=HoGRN split={} dir=avg filtered=1 tie_aware=1 full_entity=1 mrr={:.5f} h1={:.5f} h3={:.5f} h10={:.5f}'.format(
                    log_split,
                    results['mrr'],
                    results['hits@1'],
                    results['hits@3'],
                    results['hits@10']
                )
            )

        return results

    def log_final_metrics(self, results, split_label='holdout'):
        self.logger.info(
            'FINAL_EVAL_METRICS baseline=HoGRN model={} dataset={} split={} mrr={:.5f} h1={:.5f} h3={:.5f} h10={:.5f}'.format(
                self.p.score_func,
                self.p.dataset,
                split_label,
=======
        self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
        self.logger.info('[Epoch {} {}]: MR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mr'], results['right_mr'], results['mr']))
        # for k in range(10):
        # 	self.logger.info('[Epoch {} {}]: Hit@{}: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, k+1, results['left_hits@{}'.format(k+1)], results['right_hits@{}'.format(k+1)], results['hits@{}'.format(k+1)]))
        if split == 'test':
            for k in range(10):
                self.logger.info('[Epoch {} {}]: Hit@{}: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, k+1, results['left_hits@{}'.format(k+1)], results['right_hits@{}'.format(k+1)], results['hits@{}'.format(k+1)]))

        self.logger.info(
            'EVAL_STD model=HoGRN split={} dir=avg filtered=1 tie_aware=1 full_entity=1 mrr={:.5f} h1={:.5f} h3={:.5f} h10={:.5f}'.format(
                split,
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
                results['mrr'],
                results['hits@1'],
                results['hits@3'],
                results['hits@10']
            )
        )

<<<<<<< HEAD
    def append_metrics_csv(self, results, seconds):
        output_root = os.environ.get("SPARSEKGC_OUTPUT_DIR")
        if output_root:
            timing_dir = Path(output_root)
        else:
            timing_dir = Path("timings")
        timing_dir.mkdir(parents=True, exist_ok=True)
        path = timing_dir / "hogrn_metrics.csv"
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Dataset", "Model", "MRR", "Hits@1", "Hits@3", "Hits@10", "seconds"])
            writer.writerow([
                self.p.dataset,
                self.p.score_func,
                f"{results['mrr']:.5f}",
                f"{results['hits@1']:.5f}",
                f"{results['hits@3']:.5f}",
                f"{results['hits@10']:.5f}",
                f"{seconds:.3f}",
            ])
=======
        return results
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode
        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred, _ = self.model.forward(sub, rel)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                # 修正：使用 bool 遮罩並避免覆蓋目標分數
                pred = pred.masked_fill(label.bool(), -1e7)
                pred[b_range, obj] = target_pred
                greater = (pred > target_pred.unsqueeze(1)).sum(dim=1).float()
                equal = (pred == target_pred.unsqueeze(1)).sum(dim=1).float()
                ranks = greater + (equal + 1.0) / 2.0

                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

                # if step % 100 == 0:
                # 	self.logger.info('[{}, {} Step {}]'.format(split.title(), mode.title(), step))

        return results
    
    def dump_error_cases(self, split, topk=10):
        """
        逐筆輸出排名與錯誤案例到 checkpoints/{dataset}_error_cases/{split}_error_cases.csv 與 .pkl
        CSV 欄位: index, split, mode, head, relation, tail, head_id, rel_id, tail_id, rank, target_score, top1_ent, top1_score, topk_pred
        pkl: (top1_error_indices, top10_error_indices)
        """
        save_dir = os.path.join('./checkpoints', f'{self.p.dataset}_error_cases')
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f'{split}_error_cases.csv')
        pkl_path = os.path.join(save_dir, f'{split}_error_cases.pkl')

        cases = []
        for mode in ('tail_batch', 'head_batch'):
            key = f"{split}_{'tail' if mode=='tail_batch' else 'head'}"
            loader = self.data_iter[key]
            with torch.no_grad():
                for batch in loader:
                    sub, rel, obj, label = self.read_batch(batch, split)
                    pred, _ = self.model.forward(sub, rel)
                    bsz = pred.size(0)
                    b_range = torch.arange(bsz, device=self.device)

                    # 目標分數與遮罩（filtered setting）
                    target_pred = pred[b_range, obj]
                    pred = pred.masked_fill(label.bool(), -1e7)
                    pred[b_range, obj] = target_pred

                    # 排名與 top-k
                    ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                    top1_idx = torch.argmax(pred, dim=1)
                    topk_idx = torch.topk(pred, k=min(topk, pred.size(1)), dim=1).indices

                    # 還原原始 (h, r, t)
                    for i in range(bsz):
                        r = int(rel[i].item())
                        h_id = int(sub[i].item())
                        t_id = int(obj[i].item())
                        # head 模式的 triple 事先被轉成 (t, r+num_rel, h)，需轉回原始
                        if mode == 'head_batch':
                            orig_r = r - self.p.num_rel
                            orig_h, orig_t = t_id, h_id
                            orig_r_id = r - self.p.num_rel
                        else:
                            orig_r = r
                            orig_h, orig_t = h_id, t_id
                            orig_r_id = r

                        rank_i = int(ranks[i].item())
                        top1_i = int(top1_idx[i].item())
                        topk_i = [int(x) for x in topk_idx[i].tolist()]
                        cases.append({
                            'split': split,
                            'mode': 'head' if mode=='head_batch' else 'tail',
                            'head_id': orig_h,
                            'rel_id':  orig_r_id,
                            'tail_id': orig_t,
                            'head': self.id2ent[orig_h],
                            'relation': self.id2rel[orig_r_id],
                            'tail': self.id2ent[orig_t],
                            'rank': rank_i,
                            'target_score': float(target_pred[i].item()),
                            'top1_ent': self.id2ent[top1_i],
                            'top1_score': float(pred[i, top1_i].item()),
                            'topk_pred': ';'.join(self.id2ent[e] for e in topk_i),
                        })

        # 寫 CSV
        fieldnames = ['index','split','mode','head','relation','tail','head_id','rel_id','tail_id','rank','target_score','top1_ent','top1_score','topk_pred']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for idx, row in enumerate(cases):
                row_out = dict(index=idx, **row)
                w.writerow(row_out)

        # 寫 PKL（回傳索引集合）
        top1_errors = [i for i, r in enumerate(cases) if r['rank'] != 1]
        top10_errors = [i for i, r in enumerate(cases) if r['rank'] > 10]
        with open(pkl_path, 'wb') as f:
            pickle.dump((top1_errors, top10_errors), f)

        self.logger.info(f"[{split}] error cases saved to: {csv_path} (rows={len(cases)})")

    def run_epoch(self, epoch, val_mrr = 0):
        """
        Function to run one epoch of training
        """
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, obj, label = self.read_batch(batch, 'train')

            pred, cor	= self.model.forward(sub, rel)
            loss	= self.model.loss(pred, label)

            if self.p.sim_decay > 0:
                loss += self.p.sim_decay * cor

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            # if step % 100 == 0:
            # 	self.logger.info('[E:{}| {}]: Train Loss:{:.5}'.format(epoch, step, np.mean(losses)))

        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def fit(self):
        """
        Function to run training and evaluation of model.
        """
<<<<<<< HEAD
        run_start = getattr(self.p, '_run_start_time', time.perf_counter())
=======
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join('./checkpoints', self.p.name)

        # 如果只有 dump_errors 而沒有 restore，嘗試從資料集資料夾載入模型
        if getattr(self.p, 'dump_errors', False) and not self.p.restore:
            # 嘗試從 ./checkpoints/{dataset}/{dataset}_hogrn_pattern_eval 載入
            dataset_checkpoint_dir = os.path.join('./checkpoints', self.p.dataset)
            dataset_checkpoint_file = os.path.join(dataset_checkpoint_dir, f'{self.p.dataset}_hogrn_pattern_eval')
            
            if os.path.exists(dataset_checkpoint_file):
                self.logger.info(f'Inference mode: Loading model from {dataset_checkpoint_file}')
                
                # 載入 checkpoint 中保存的參數
                saved_args = self.load_checkpoint_args(dataset_checkpoint_file)
                if saved_args:
                    self.logger.info('Reconstructing model with saved arguments')
                    # 更新當前參數（保留 dataset, gpu, dump_errors 等 inference 相關參數）
                    inference_params = ['dataset', 'gpu', 'dump_errors', 'topk', 'name', 'log_dir', 'config_dir']
                    for key, value in saved_args.items():
                        if key not in inference_params:
                            setattr(self.p, key, value)
                    
                    # 重新初始化模型和優化器
                    self.model = self.add_model(self.p.model, self.p.score_func)
                    self.optimizer = self.add_optimizer(self.model.parameters())
                
                self.load_model(dataset_checkpoint_file)
                self.logger.info('Successfully loaded dataset checkpoint')
                
                test_results = self.evaluate('test', self.best_epoch)
                self.logger.info('Test Avg MRR: {:.5}'.format(test_results['mrr']))
                self.dump_error_cases('test', topk=getattr(self.p, 'topk', 10))  
                self.dump_error_cases('valid', topk=getattr(self.p, 'topk', 10))
                self.dump_error_cases('train', topk=getattr(self.p, 'topk', 10))
                return
            else:
                self.logger.warning(f'Dataset checkpoint not found at {dataset_checkpoint_file}, proceeding with training')

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        # 如果同時有 restore 和 dump_errors，直接進行 inference 模式
        if self.p.restore and getattr(self.p, 'dump_errors', False):
            self.logger.info('Inference mode: skipping training, only evaluating and dumping errors')
<<<<<<< HEAD
            test_results = self.evaluate('test', self.best_epoch, emit_detail=False)
            self.logger.info('Final Avg MRR: {:.5}'.format(test_results['mrr']))
            self.log_final_metrics(test_results)
            self.append_metrics_csv(test_results, time.perf_counter() - run_start)
=======
            test_results = self.evaluate('test', self.best_epoch)
            self.logger.info('Test Avg MRR: {:.5}'.format(test_results['mrr']))
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
            self.dump_error_cases('test', topk=getattr(self.p, 'topk', 10))  
            self.dump_error_cases('valid', topk=getattr(self.p, 'topk', 10))
            self.dump_error_cases('train', topk=getattr(self.p, 'topk', 10))
            return

        kill_cnt = 0
        # for epoch in range(1):
        for epoch in range(self.p.max_epochs):
            print("########")
            t0 = time.time()
            train_loss  = self.run_epoch(epoch, val_mrr)
            print("Time cost in one epoch for training: {:.4f}s".format((time.time()-t0)))

            val_results = self.evaluate('valid', epoch)
            
            if val_results['mrr'] > self.best_val_mrr:
                self.best_val	   = val_results
                self.best_val_mrr  = val_results['mrr']
                self.best_epoch	   = epoch
                self.save_model(save_path)
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.p.gamma > 5:
                    self.p.gamma -= 5 
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                if kill_cnt > 25: 
                    self.logger.info("Early Stopping!!")
                    break

            self.logger.info('[Epoch {}]: Training Loss: {:.5}, Best Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))
            # print("Total time cost in one epoch: {:.4f}s".format((time.time()-t0)/60))

<<<<<<< HEAD
        self.logger.info('Loading best model, evaluating on holdout data')
        self.load_model(save_path)
        test_results = self.evaluate('test', epoch, emit_detail=False)
        self.logger.info('Final Avg MRR: {:.5}'.format(test_results['mrr']))
        self.log_final_metrics(test_results)
        self.append_metrics_csv(test_results, time.perf_counter() - run_start)
=======
        self.logger.info('Loading best model, Evaluating on Test data')
        self.load_model(save_path)
        test_results = self.evaluate('test', epoch)
        self.logger.info('Test Avg MRR: {:.5}'.format(test_results['mrr']))
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
        
        if getattr(self.p, 'dump_errors', False):
            self.dump_error_cases('test', topk=getattr(self.p, 'topk', 10))  
            self.dump_error_cases('valid', topk=getattr(self.p, 'topk', 10))
            self.dump_error_cases('train', topk=getattr(self.p, 'topk', 10))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name',		dest='name',		default='testrun',		help='Set run name for saving/restoring models')
    parser.add_argument('-data',		dest='dataset',		default='FB15K-237-10',	help='Dataset to use.')
    parser.add_argument('-model',		dest='model',		default='hogrn',		help='Model Name')
    parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
    parser.add_argument('-opn',         dest='opn',			default='mult',			help='Composition Operation to be used in HoGRN')

    parser.add_argument('-batch',       dest='batch_size',	type=int, 	default=128,	help='Batch size')
    parser.add_argument('-epoch',		dest='max_epochs',	type=int,	default=9999,  	help='Number of epochs')
    parser.add_argument('-gamma',		dest='gamma',		type=float,	default=40,		help='Margin')
    parser.add_argument('-gpu',			type=str,			default='0',				help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    
    parser.add_argument('-l2',			type=float,	default=0,		help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',			type=float,	default=0.001,	help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth',	type=float,	default=0.1,	help='Label Smoothing')
    parser.add_argument('-num_workers',	type=int,	default=2,		help='Number of processes to construct batches')
    parser.add_argument('-seed',		type=int,	default=41504, 	help='Seed for randomization')

    parser.add_argument('-restore',     dest='restore',		action='store_true',	help='Restore from the previously saved model')
    parser.add_argument('-bias',		dest='bias',		action='store_true',	help='Whether to use bias in the model')

    parser.add_argument('-rel_reason', 	dest='rel_reason',	action='store_true',	help='Whether to optimize the relation representation by relation reasoning')
    parser.add_argument('-pre_reason', 	dest='pre_reason',	action='store_true',	help='Whether to use the relation reasoning firstly')
    parser.add_argument('-reason_type', dest='reason_type',	default='mixdrop',		help='Relation Reason Operation to be used in HoGRN')
    parser.add_argument('-act_type', 	dest='act_type',	default='tanh',			help='Activation funtion to be used in HoGRN')
    parser.add_argument('-rel_norm', 	dest='rel_norm',	action='store_true',	help='Whether to optimize the relation representation by normalization')
    
    parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=100,   	type=int, 	help='Number of hidden units in GCN') 
    parser.add_argument('-embed_dim',	dest='embed_dim', 	default=100,   	type=int, 	help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop',	dest='dropout', 	default=0,  	type=float,	help='Dropout to use in GCN Layer') 
    parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0,  	type=float,	help='Dropout after GCN')
    parser.add_argument('-relmix_dim',	dest='relmix_dim',	default=200,	type=int,	help='Number of hidden units in inter-relation learning')
    parser.add_argument('-chamix_dim',	dest='chamix_dim', 	default=200,  	type=int, 	help='Number of hidden units in intra-relation learning') 
    parser.add_argument('-rel_mask',  	dest='rel_mask', 	default=0,  	type=float,	help='Dropout in inter-relation learning')
    parser.add_argument('-chan_drop',  	dest='chan_drop', 	default=0,  	type=float,	help='Dropout in intra-relation learning')
    parser.add_argument('-edge_drop',  	dest='edge_drop', 	default=0,  	type=float,	help='Dropout in edge')

    # Relational contrastive loss
    parser.add_argument('-temperature', dest='temperature', default=1,  	type=float,	help='temperature coefficient')
    parser.add_argument('-sim_decay',	dest='sim_decay',	default=0,		type=float, help='Regularization weight for independence modeling')
    parser.add_argument('-rel_drop',  	dest='rel_drop', 	default=0,  	type=float,	help='Dropout for generate positive relation')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
    parser.add_argument('-k_w',	  		dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
    parser.add_argument('-k_h',	  		dest='k_h', 		default=10,   	type=int, 	help='ConvE: k_h')
    parser.add_argument('-num_filt',  	dest='num_filt', 	default=32,   	type=int, 	help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=3,   	type=int, 	help='ConvE: Kernel size to use')

    parser.add_argument('-logdir',		dest='log_dir',		default='./log/',		help='Log directory')
    parser.add_argument('-config',		dest='config_dir',	default='./config/',	help='Config directory')
    
    parser.add_argument('-dump_errors', action='store_true', help='Dump error cases (CSV/PKL) for valid/test')
    parser.add_argument('-topk',        type=int, default=10, help='Top-K predictions to export for each case')
    args = parser.parse_args()

    if not args.restore:
        ts = time.strftime('%Y-%m-%d_%H-%M-%S')
        args.name = f"{args.name}_{ts}"
    os.makedirs(args.log_dir, exist_ok=True)


    # set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)
    
<<<<<<< HEAD
    args._run_start_time = time.perf_counter()
=======
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
    model = Runner(args)
    model.fit()
