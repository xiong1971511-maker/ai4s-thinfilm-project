"""Verify the published dataset, checkpoint, spectra, histories and candidate ranks."""
import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import torch
from ai4s_thinfilm.config import STUDY_CONFIG as C
from ai4s_thinfilm.dataset import generate_dataset, split_indices
from ai4s_thinfilm.design import generate_candidates, verify_spectra, rank_candidates
from ai4s_thinfilm.model import MLPRegressor

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='verification/submission_verification.json')
    args=parser.parse_args()
    torch.set_num_threads(1)
    checks={}
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition: raise AssertionError(name)
    with np.load('data/thinfilm_dataset.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    x,y=generate_dataset()
    data_error=float(np.max(np.abs(y-data['reflectance'])))
    check('dataset_regenerates_from_seed',np.array_equal(x,data['thicknesses_nm']) and data_error<1e-12)
    splits=split_indices(5000,C.seed,(4000,500,500))
    check('fixed_disjoint_splits_match',all(np.array_equal(s,data[k]) for s,k in zip(splits,('train_indices','validation_indices','test_indices'))))
    checkpoint=torch.load('models_convergence_linear/mlp_train_4000.pt',map_location='cpu',weights_only=True)
    model=MLPRegressor(output_activation='linear')
    model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    scaled=((x-40)/140).astype(np.float32)
    with torch.no_grad():pred=model(torch.from_numpy(scaled[splits[2]])).numpy()
    truth=y[splits[2]].astype(np.float32)
    mse=float(np.mean((pred-truth)**2));rmse=float(np.sqrt(mse));mae=float(np.mean(np.abs(pred-truth)))
    report=json.loads(Path('outputs_convergence_linear/mlp_metrics.json').read_text())
    expected=report['results_by_training_size']['4000']
    check('checkpoint_test_metrics_match',abs(mse-expected['test_mse'])<1e-9 and abs(rmse-expected['test_rmse'])<1e-7 and abs(mae-expected['test_mae'])<1e-7)
    with torch.no_grad():val=model(torch.from_numpy(scaled[splits[1]])).numpy()
    vmse=float(np.mean((val-y[splits[1]].astype(np.float32))**2))
    check('checkpoint_validation_metric_matches',abs(vmse-expected['validation_mse'])<1e-9)
    for size,record in report['results_by_training_size'].items():
        hist=np.genfromtxt(f'outputs_convergence_linear/loss_history_train_{size}.csv',delimiter=',',names=True)
        check(f'history_{size}_epochs_and_best_metric',len(hist)==record['epochs_run'] and abs(float(hist['validation_mse'][record['best_epoch']-1])-record['validation_mse'])<1e-12)
    with np.load('outputs_convergence_linear/design_screening_predictions.npz',allow_pickle=False) as z:
        candidates=z['thicknesses_nm'];saved_pred=z['predicted_reflectance']
    check('candidate_seed_and_pool_match',np.array_equal(candidates,generate_candidates()))
    check('candidates_do_not_duplicate_dataset',not (set(map(tuple,x)) & set(map(tuple,candidates))))
    with torch.no_grad():cp=model(torch.from_numpy(((candidates-40)/140).astype(np.float32))).numpy()
    inference_error=float(np.max(np.abs(cp-saved_pred)))
    check('candidate_predictions_match_checkpoint',inference_error<2e-6)
    tmm=verify_spectra(candidates)
    with np.load('outputs_convergence_linear/global_tmm_audit/linear_candidate_full_tmm.npz',allow_pickle=False) as z:
        tmm_error=float(np.max(np.abs(tmm-z['tmm_reflectance_spectra'])))
        check('all_10000_candidate_spectra_match_tmm',tmm_error<1e-12)
        check('saved_full_pool_thicknesses_match',np.array_equal(candidates,z['thicknesses_nm']))
    target=30
    minimum=int(np.argmin(tmm[:,target]));maximum=int(np.argmax(tmm[:,target]))
    check('global_extrema_match_paper',minimum==3405 and maximum==2256)
    overlaps={}
    for objective in ('min','max'):
        proxy_ids=rank_candidates(saved_pred,target,objective)[:10]
        physical_ids=rank_candidates(tmm,target,objective)[:5]
        overlaps[objective]=len(set(proxy_ids)&set(physical_ids))
        rows=list(csv.DictReader(open(f'outputs_convergence_linear/design_top10_{objective}.csv',encoding='utf-8')))
        check(f'{objective}_shortlist_ids_match',np.array_equal(proxy_ids,[int(r['candidate_index']) for r in rows]))
        check(f'{objective}_shortlist_values_match',all(abs(float(r['tmm_reflectance_at_target'])-tmm[int(r['candidate_index']),target])<1e-12 for r in rows))
    check('shortlist_overlap_matches_paper',overlaps=={'min':0,'max':3})
    high=int(np.argmax(np.sqrt(np.mean((pred-truth)**2,axis=1))))
    check('highest_error_case_matches_paper',high==312 and int(splits[2][high])==3477)
    case=np.genfromtxt('paper/heldout_test_case_row_3477_spectra.csv',delimiter=',',names=True)
    case_error=float(np.max(np.abs(case['linear_prediction']-pred[high])))
    check('failure_case_csv_matches_checkpoint',case_error<2e-6)
    result={'status':'passed','environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},'checks':checks,'test_mse':mse,'test_rmse':rmse,'test_mae':mae,'validation_mse':vmse,'dataset_tmm_max_abs_error':data_error,'candidate_tmm_max_abs_error':tmm_error,'candidate_inference_max_abs_error':inference_error,'failure_case_prediction_max_abs_error':case_error,'global_minimum':{'id':minimum,'reflectance':float(tmm[minimum,target])},'global_maximum':{'id':maximum,'reflectance':float(tmm[maximum,target])},'shortlist_overlaps':overlaps}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
