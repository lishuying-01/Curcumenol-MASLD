import pandas as pd



from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import SaltRemover

def calculate_molecular_features(df):
    """
    从SMILES计算分子特征
    包括：分子描述符 + 分子指纹
    """
    print("\n正在计算分子特征...")
    
    features_list = []
    valid_indices = []
    failed_smiles = []
    
    # 初始化盐移除器
    remover = SaltRemover.SaltRemover()
    
    for idx, row in df.iterrows():
        smiles = str(row['Smiles'])
        
        try:
            # 1. 解析SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed_smiles.append(smiles)
                continue
            
            # 2. 去除盐
            mol = remover.StripMol(mol)
            
            # 3. 计算分子描述符（物理化学性质）
            descriptors = {
                # 基本性质
                'mol_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                
                # 氢键
                'num_h_donors': Descriptors.NumHDonors(mol),
                'num_h_acceptors': Descriptors.NumHAcceptors(mol),
                
                # 键和环
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_rings': Descriptors.RingCount(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                
                # 电荷和极性
                'formal_charge': Chem.GetFormalCharge(mol),
                'fraction_csp3': Descriptors.FractionCSP3(mol),
                
                # 其他重要描述符
                'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
                'num_heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'num_valence_electrons': Descriptors.NumValenceElectrons(mol),
            }
            
            # 4. 计算分子指纹（结构指纹）
            # 摩根指纹
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
            for i, bit in enumerate(morgan_fp):
                descriptors[f'morgan_bit_{i}'] = bit
            
            # MACCS指纹（可选）
            # from rdkit.Chem import MACCSkeys
            # maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            # for i, bit in enumerate(maccs_fp):
            #     descriptors[f'maccs_bit_{i}'] = bit
            
            features_list.append(descriptors)
            valid_indices.append(idx)
            
        except Exception as e:
            failed_smiles.append(smiles)
            continue
    
    # 创建特征DataFrame
    features_df = pd.DataFrame(features_list)
    
    # 合并回原始数据
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    df_with_features = pd.concat([df_valid, features_df], axis=1)
    
    print(f"✅ 特征计算完成:")
    print(f"   成功: {len(df_with_features)}/{len(df)} 个分子")
    print(f"   失败: {len(failed_smiles)} 个分子")
    
    if failed_smiles:
        print(f"   失败的SMILES示例: {failed_smiles[:5]}")
    
    return df_with_features


def train_process(postive_path, negtive_path, save_path):
    df = pd.read_excel(postive_path)
    print(df.columns)
    post= df[["Smiles"]]

    df = pd.read_excel(negtive_path)
    print(df.columns)
    neg = df[["Smiles"]]
    df_post = calculate_molecular_features(post)
    df_neg = calculate_molecular_features(neg)
    df_post["label"] = 1
    df_neg["label"] = 0
    print()
    all_df = pd.concat([df_post, df_neg])
    all_df.to_excel(save_path, index=False)


def predict_process(path, save_path):
    df = pd.read_excel(path)
    print(df.shape)
    # post = df[["smile"]].rename(columns={"smile": "Smiles"})
    # pre_data = post["Smiles"].tolist()
    # train_data = pd.read_excel("all_data.xlsx")["Smiles"].tolist()
    # print()
    # pre_data = [one for one in pre_data if one not in train_data]
    # post = pd.DataFrame({"Smiles":pre_data})
    df_post = calculate_molecular_features(df)
    df_post["label"] = 0
    df_post.to_excel(save_path, index=False)


if __name__ == "__main__":
    # train data process
    # postive_path = "data/AMPK activitor/postive.xlsx"
    # negtive_path = "data/inhibitor/negtive.xlsx"
    # save_path = "data/train_data.xlsx"
    # train_process(postive_path, negtive_path, save_path)

    # predict data process 
    pred_data_path = "data/raw_predict_data.xlsx"
    pred_save_path = "data/predict_data.xlsx"
    predict_process(pred_data_path, pred_save_path)
