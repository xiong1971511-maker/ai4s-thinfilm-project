"""Reproduce all three final article figures using the published Linear artifacts."""
import json
from pathlib import Path
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from ai4s_thinfilm.model import MLPRegressor

def main():
    out=Path('paper/figures');out.mkdir(parents=True,exist_ok=True)
    font_path=Path(__file__).resolve().parents[1]/'assets/fonts/AI4SFigureCJK-Regular.otf'
    font_manager.fontManager.addfont(str(font_path))
    cjk_name=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.size':8.5,'axes.titlesize':9,'axes.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path','font.family':['DejaVu Sans',cjk_name]})
    blue='#235789';teal='#008477';orange='#d47a00'
    z=np.load('data/thinfilm_dataset.npz',allow_pickle=False);w=z['wavelengths_nm'];test=z['test_indices']
    ck=torch.load('models_convergence_linear/mlp_train_4000.pt',weights_only=True,map_location='cpu')
    m=MLPRegressor(output_activation='linear');m.load_state_dict(ck['model_state_dict']);m.eval();torch.set_num_threads(1)
    with torch.no_grad():pred=m(torch.from_numpy(((z['thicknesses_nm'][test]-40)/140).astype(np.float32))).numpy()
    truth=z['reflectance'][test]
    metrics=json.loads(Path('outputs_convergence_linear/mlp_metrics.json').read_text())['results_by_training_size']
    hist=np.genfromtxt('outputs_convergence_linear/loss_history_train_4000.csv',delimiter=',',names=True)
    pool=np.load('outputs_convergence_linear/global_tmm_audit/linear_candidate_full_tmm.npz',allow_pickle=False)
    p=pool['linear_predicted_reflectance'];t=pool['tmm_reflectance_spectra'];target=30
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=260,bbox_inches='tight',facecolor='white')
        fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
    fig=plt.figure(figsize=(9,6.0),layout='constrained');gs=fig.add_gridspec(3,2,height_ratios=[1.35,0.9,1.5])
    a=fig.add_subplot(gs[0,:]);a.set_axis_off();a.set_title('(a) 膜系结构与 TMM 数据生成\nOptical stack and TMM data generation',loc='left',fontweight='bold')
    labels=['空气\nAir','高折射率\nH','低折射率\nL','高折射率\nH','低折射率\nL','玻璃\nGlass'];colors=['#eef3fa','#f4d5cb','#cfe2f3','#f4d5cb','#cfe2f3','#dfe6e8']
    for j,(s,c) in enumerate(zip(labels,colors)):
        a.add_patch(FancyBboxPatch((.02+j*.112,.56),.095,.23,boxstyle='round,pad=0.005',facecolor=c,edgecolor='#91a6bb'))
        a.text(.068+j*.112,.67,s,ha='center',va='center')
    a.text(.34,.38,'nH = 2.30; nL = 1.45; ns = 1.52',ha='center')
    a.annotate('',xy=(.79,.67),xytext=(.69,.67),arrowprops={'arrowstyle':'->','color':blue})
    a.text(.89,.67,'TMM\n4 层膜厚 / Thicknesses\n41 点反射谱 / R(λ)',ha='center',va='center',bbox={'boxstyle':'round','fc':'#e6f4f2','ec':teal})
    a.text(.02,.06,'设种采样 → TMM → 固定划分 → MLP → 新候选 → TMM\nSeeded samples → TMM → 4000 / 500 / 500 → MLP → 10000 candidates → TMM',color=blue)
    b=fig.add_subplot(gs[1,:]);b.set_axis_off();b.set_title('(b) MLP 网络结构与训练设置\nMLP architecture and training settings',loc='left',fontweight='bold')
    for j,s in enumerate(['4 输入\nInputs','128 ReLU','128 ReLU','64 ReLU','41 线性输出\nLinear']):
        b.text(.10+j*.20,.56,s,ha='center',bbox={'boxstyle':'round,pad=0.55','fc':'#eaf1f8','ec':blue})
        if j<4:b.annotate('',xy=(.24+j*.20,.57),xytext=(.17+j*.20,.57),arrowprops={'arrowstyle':'->'})
    b.text(.5,.10,'输入缩放 / Input (d − 40) / 140 | Adam 0.001 | 批大小 / Batch 128 | MSE | 上限 / Cap 2500',ha='center')
    c=fig.add_subplot(gs[2,0]);sizes=[500,1000,2000,4000];c.plot(sizes,[metrics[str(n)]['validation_mse'] for n in sizes],'o-',color=teal)
    c.set(xlabel='训练样本数 / Training samples',ylabel='验证 MSE / Validation MSE',yscale='log',title='(c) 训练规模实验\nTraining-size experiment');c.grid(alpha=.2)
    d=fig.add_subplot(gs[2,1]);d.semilogy(hist['epoch'],hist['training_mse'],label='训练 Training',color=blue);d.semilogy(hist['epoch'],hist['validation_mse'],label='验证 Validation',color=orange);d.axvline(1538,ls=':',color='gray');d.set(xlabel='训练轮次 / Epoch',ylabel='MSE',title='(d) 主模型损失曲线\nMain-model loss histories');d.legend();d.grid(alpha=.2)
    save(fig,'figure_1_workflow_and_tmm')
    fig=plt.figure(figsize=(8.5,8.0),layout='constrained');gs=fig.add_gridspec(4,2,height_ratios=[1.3,1.3,.8,1.2])
    a=fig.add_subplot(gs[0,:]);a.scatter(p[:,target],t[:,target],s=3,alpha=.25,c=blue,rasterized=True)
    a.plot([-.03,.7],[-.03,.7],ls=':',color='gray');a.set(xlabel='MLP 预测 / Predicted R(700 nm)',ylabel='TMM 真值 / True R(700 nm)',title='(a) 固定 10000 候选池\nFixed 10000-candidate pool');a.grid(alpha=.2)
    for j,(idx,col,label) in enumerate([(3405,teal,'全池最小 / Pool minimum'),(2256,orange,'全池最大 / Pool maximum')]):
        ax=fig.add_subplot(gs[1,j]);ax.plot(w,t[idx],c=col,label='真值 TMM');ax.plot(w,p[idx],ls='--',c=blue,label='预测 MLP');ax.axvline(700,ls=':',c='gray');ax.set(xlabel='波长 / Wavelength (nm)',ylabel='反射率 / Reflectance',title=f'{label}\nID {idx}');ax.legend(fontsize=8);ax.grid(alpha=.2)
    short={o:list(csv.DictReader(open(f'outputs_convergence_linear/design_top10_{o}.csv'))) for o in ('min','max')}
    ax=fig.add_subplot(gs[2,:])
    for o,c in [('min',teal),('max',orange)]:
        ids=[int(r['candidate_index']) for r in short[o]];ax.scatter(p[ids,target],t[ids,target],label=('最小化 / Min' if o=='min' else '最大化 / Max'),c=c,s=22)
    ax.plot([-.03,.7],[-.03,.7],ls=':',c='gray');ax.set(xlabel='MLP 预测 / Predicted R(700 nm)',ylabel='TMM 真值 / True R(700 nm)',title='(b) MLP 前 10 候选的物理验证\nTMM verification of MLP Top 10');ax.legend(fontsize=8);ax.grid(alpha=.2)
    for j,o in enumerate(('min','max')):
        idx=int(min(short[o],key=lambda r:int(r['tmm_rank_within_mlp_top10']))['candidate_index']);ax=fig.add_subplot(gs[3,j]);ax.plot(w,t[idx],c=teal if j==0 else orange,label='真值 TMM');ax.plot(w,p[idx],ls='--',c=blue,label='预测 MLP');ax.axvline(700,ls=':',c='gray');ax.set(xlabel='波长 / Wavelength (nm)',ylabel='反射率 / Reflectance',title=f"{'短名单最小 / Shortlist min' if o=='min' else '短名单最大 / Shortlist max'}\nID {idx}");ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'figure_2_prediction_and_design')
    fig=plt.figure(figsize=(8.5,8.0),layout='constrained');gs=fig.add_gridspec(4,3,height_ratios=[1.25,1.7,.8,1.2])
    for j in range(3):
        ax=fig.add_subplot(gs[0,j]);ax.plot(w,truth[j],c=blue,label='真值 TMM');ax.plot(w,pred[j],ls='--',c=orange,label='预测 MLP');ax.set(xlabel='波长 / Wavelength (nm)',ylabel='R',title=f'(a{j+1}) 测试样本 / Test\nRow {test[j]}');ax.legend(fontsize=7);ax.grid(alpha=.2)
    high=int(np.argmax(np.sqrt(np.mean((pred-truth)**2,axis=1))))
    ax=fig.add_subplot(gs[1,:]);ax.plot(w,truth[high],c=blue,label='真值 TMM');ax.plot(w,pred[high],ls='--',c=orange,label='预测 MLP');ax.axvline(420,ls=':',c='gray');ax.set(xlabel='波长 / Wavelength (nm)',ylabel='反射率 / Reflectance',title=f'(b) 最高误差测试案例 / Highest-RMSE test case\nItem {high+1}; row {test[high]}');ax.legend();ax.grid(alpha=.2)
    ax=fig.add_subplot(gs[2,:]);ax.plot(w,np.abs(pred[high]-truth[high]),c='#bc3030');ax.set(xlabel='波长 / Wavelength (nm)',ylabel='绝对误差 / Absolute error');ax.grid(alpha=.2)
    ax=fig.add_subplot(gs[3,:2]);ax.semilogy(hist['epoch'],hist['training_mse'],label='训练 Training',c=blue);ax.semilogy(hist['epoch'],hist['validation_mse'],label='验证 Validation',c=orange);ax.axvline(1538,ls=':',c='gray');ax.set(xlabel='训练轮次 / Epoch',ylabel='MSE',title='(c) 线性主模型训练\nLinear main-model training');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=fig.add_subplot(gs[3,2]);ax.plot(sizes,[metrics[str(n)]['test_rmse'] for n in sizes],'o-',c=teal);ax.set(xlabel='训练样本数 / Training samples',ylabel='测试 RMSE / Test RMSE',title='(d) 数据量影响\nData-size effect');ax.grid(alpha=.2)
    save(fig,'figure_3_training_and_failure')
    manifest={'status':'success','model':'published Linear checkpoint','figures':['figure_1_workflow_and_tmm','figure_2_prediction_and_design','figure_3_training_and_failure'],'seed':270256,'design_seed':270257,'target_wavelength_nm':700,'language':'zh-en','font':'assets/fonts/AI4SFigureCJK-Regular.otf','sources':['data/thinfilm_dataset.npz','models_convergence_linear/mlp_train_4000.pt','outputs_convergence_linear/mlp_metrics.json','outputs_convergence_linear/loss_history_train_4000.csv','outputs_convergence_linear/global_tmm_audit/linear_candidate_full_tmm.npz','outputs_convergence_linear/design_top10_min.csv','outputs_convergence_linear/design_top10_max.csv']}
    (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
