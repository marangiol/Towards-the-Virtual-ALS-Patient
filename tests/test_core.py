import numpy as np
from src.connectome import load_connectome
from src.avalanches import directional_atm
from src.atm_metrics import metrics_from_matrix
from src.simulation import make_design

def test_demo():
    a=load_connectome('configs/demo.json');b=load_connectome('configs/demo.json')
    np.testing.assert_array_equal(a['W'],b['W'])
    assert len(a['frontal_indices'])==34 and len(a['no_frontal_indices'])==46

def test_avalanche_definition():
    # 0->1 transition in first event; isolated one-bin activity is ignored.
    binary=np.array([[1,0,0,1,0],[0,1,0,0,0]],dtype=bool)
    matrix,events=directional_atm(binary)
    assert events==1
    np.testing.assert_array_equal(matrix,[[0,1],[0,0]])

def test_directional_metrics():
    values=metrics_from_matrix(np.array([[0.,1.],[.25,0.]]))
    assert values['ATM_mean_mat']==1 and values['ATM_mean_lower']==.25
    np.testing.assert_allclose(values['ATM_norm'],np.sqrt(1.0625))

def test_design():
    grid=make_design('regional',100,True)
    assert len(grid)==100 and grid.G.nunique()==1
    assert grid.G.iloc[0]==1.89
    bank=make_design('regional',100,False)
    assert bank.G.nunique()>1 and bank.noise_seed.is_unique
